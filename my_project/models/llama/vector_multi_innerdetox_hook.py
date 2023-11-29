from functools import partial
from mmengine import Registry
import torch
import torch.nn.functional as F

InnerDetoxHook = Registry('innerdetox_hook')


@InnerDetoxHook.register_module()
class BaseInnerDetoxHook():
    def __init__(self, norm_exp=0, neg_sim_exp=0, renorm=False, vector_method='mean'):
        self.mem = dict()
        self.hook_handles = dict()
        self.norm_exp = norm_exp
        self.neg_sim_exp = neg_sim_exp
        self.renorm = renorm
        self.vector_method = vector_method
        self.lamda = 1
        self.batch_ids = None
        self.module_nm_to_ids = dict()  # 用于判断生成第几层的编码

    def read_hook(self, module, input, output, module_name=None, prompt_end_indices=None, neg_num_per_sample=None, ids=None, module_names_specialized=None):
        quarter_bsz = output.shape[0] // (neg_num_per_sample + 1)  # 最开始的 1/(neg_num_per_sample+1) 部分是 正示例， 剩下的都是负的
        batch, n_head, seq_len, dim = output.shape
        # neg - pos
        if self.mem.get(module_name, None) is None:
            self.mem[module_name] = dict()

        neg_vector_all = dict()
        delta_all = dict()

        for i in range(neg_num_per_sample):
            neg_vector_all[str(f'neg_{i}')] = output[quarter_bsz * (i + 1):quarter_bsz * (i + 2), :, -1:, :].detach()

        pos_vector = output[:quarter_bsz, :, -1:, :].detach()

        # TODO: 方法1. 任务向量的平均融合
        if self.vector_method == 'mean':
            for key in neg_vector_all.keys():
                delta_all[key] = neg_vector_all[key] - pos_vector
            delta = torch.stack([delta_all[key] for key in delta_all.keys()]).mean(dim=0, keepdim=False)

        # TODO: 方法2. 任务向量的过滤融合
        elif 'merging' in self.vector_method:
            for key in neg_vector_all.keys():
                delta_all[key] = (neg_vector_all[key] - pos_vector).reshape(quarter_bsz, n_head * dim)
            delta_0 = torch.transpose(torch.stack([delta_all[key] for key in delta_all.keys()]), 0,
                                      1)  # n_neg, batch, n_head*dim--> batch, n_neg, n_head*dim
            operation_func = self.vector_method[len('merging'):]  # 默认 merging topk 50
            delta = self.merge_method(func=operation_func, all_checks=delta_0, batch=quarter_bsz, n_head=n_head, dim=dim)

        self.mem[module_name]['delta'] =self.lamda * delta  # batch, n_head, 1, dim
        # if self.mem[module_name].get('neg_end', None) is None:
        #     pass

        # else:
        #     raise ValueError('prompt_end_indices is None')
        # print(f"计算结束: batch-{self.batch_ids} module_name-{module_name}")

        if self.batch_ids != ids:
            for key in module_names_specialized:
                self.module_nm_to_ids[key] = None
            self.module_nm_to_ids[module_name] = 1
            self.batch_ids = ids

            h, n, d = output.shape[1:]
            neg_end_indices = prompt_end_indices[quarter_bsz:, None, None, None].expand(-1, h, 1, d).to(output.device)
            all_neg_end = torch.gather(output[quarter_bsz:, :, :, :], dim=2, index=neg_end_indices).detach()
            all_neg_end_org = all_neg_end.view(all_neg_end.size(0) // (neg_num_per_sample) , neg_num_per_sample, n_head, 1, d)
            # b = all_neg_end_org.cpu().numpy()
            all_neg_end_mean = all_neg_end_org.mean(dim=1)
            self.mem[module_name]['neg_end'] = all_neg_end_mean
                # torch.gather(output[quarter_bsz:, :, :, :], dim=2, index=neg_end_indices)[:1, :, :, :].detach()  # self.mem[module_name]['neg_end'] : 1, n_head, 1, dim
        elif self.batch_ids == ids and self.module_nm_to_ids.get(module_name, None) is None:  # 从第二层开始
            self.module_nm_to_ids[module_name] = 1

            h, n, d = output.shape[1:]
            neg_end_indices = prompt_end_indices[quarter_bsz:, None, None, None].expand(-1, h, 1, d).to(output.device)
            all_neg_end = torch.gather(output[quarter_bsz:, :, :, :], dim=2, index=neg_end_indices).detach()
            all_neg_end_org = all_neg_end.view(all_neg_end.size(0) // (neg_num_per_sample), neg_num_per_sample, n_head, 1, d)
            # b = all_neg_end_org.cpu().numpy()
            all_neg_end_mean = all_neg_end_org.mean(dim=1)
            self.mem[module_name]['neg_end'] = all_neg_end_mean
        # else:
            # print(f"计算结束: {self.i+1}")
            # self.i = self.i+1

    def write_hook(self, module, input, output, module_name=None):
        return self.modification_fn(output, self.mem[module_name], module_name)

    def register_hooks(self, model, hook):
        for n, m in model.named_modules():
            if self.module_match_fn(n):  # todo: 定义我们在哪些位置 执行减法 做差
                handle = m.register_forward_hook(partial(hook, module_name=n))
                self.hook_handles[n] = handle

    def remove_hooks(self):
        for n in list(self.hook_handles.keys()):
            self.hook_handles[n].remove()
            self.hook_handles.pop(n)

    def module_match_fn(self, module_name):  # 这是在模型中插入的位置
        return module_name.endswith('.before_mergehead')

    def modification_fn(self, v, v_mem, module_name):
        if self.renorm:
            v_norm = v[:, :, -1:, :].norm(dim=(1, 3), keepdim=True)  # batch, n_head, 1, dim

        delta = v_mem['delta']  # batch,n_head, 1, dim
        neg_end = v_mem['neg_end']  # 1, n_head, 1, dim

        norm_scale = 1
        if self.norm_exp > 0:
            norm_scale = (1 + delta.norm(dim=-1, keepdim=True)) ** self.norm_exp

        neg_sim_scale = 1
        if self.neg_sim_exp > 0:
            neg_sim = (neg_end * v[:, :, -1:, :]).sum(dim=-1, keepdim=True) / (
                        neg_end.norm(dim=-1, keepdim=True) * v[:, :, -1:, :].norm(dim=-1, keepdim=True))
            neg_sim_scale = (1 + F.relu(neg_sim)) ** self.neg_sim_exp

        v[:, :, -1:, :] = v[:, :, -1:, :] - norm_scale * neg_sim_scale * delta

        if self.renorm:
            new_v_norm = v[:, :, -1:, :].norm(dim=(1, 3), keepdim=True)
            v[:, :, -1:, :] = v[:, :, -1:, :] * (v_norm / new_v_norm)
        return v

    def merge_method(self, func=None, all_checks=None, batch=None, n_head=None, dim=None):  # batch, n_neg, n_head*dim
        topknum, elect, agg = func.split('_')  # topk20, mass, dis-mean
        if 'topk' in topknum:
            thresh = topknum[len('topk'):]
            updated_checks = topk_values_mask(all_checks, K=int(thresh), return_mask=False)  # 除了topk位置清零，batch, n_neg, n_head*dim
        else:
            updated_checks = all_checks
        final_signs = resolve_sign(updated_checks, elect)  # elect  每一列的符号, batch, n_head*dim
        merged_tv = disjoint_merge(updated_checks, agg, final_signs)

        return merged_tv.view(batch, n_head, 1, dim)


@InnerDetoxHook.register_module()
class LayerAblationInnerDetoxHook(BaseInnerDetoxHook):
    def __init__(self, norm_exp=0, neg_sim_exp=0, renorm=False,
                 ablation_layers=[]):
        super().__init__(norm_exp, neg_sim_exp, renorm)
        self.ablation_layers = ablation_layers

    def module_match_fn(self, module_name):
        layer_idxs = module_name.split('.')
        if len(layer_idxs) < 3:
            return False
        layer_idx = layer_idxs[-3]
        if not layer_idx.isdigit():
            return False
        return module_name.endswith(f'.before_mergehead') and (int(layer_idx) not in self.ablation_layers)


def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    b, n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements, 剩下不保持的参数有哪些

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=2, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask


def normmass_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=2, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    return (Tensor.sign() * norm_fracs.abs()).sum(dim=1).sign()

def normfrac_based_sign(Tensor):
    row_norms = torch.norm(Tensor, dim=1, keepdim=True)
    norm_fracs = (Tensor ** 2) / row_norms ** 2
    max_indices = norm_fracs.argmax(dim=1).unsqueeze(1)
    selected_elements = torch.gather(Tensor, dim=1, index=max_indices)
    output = torch.sign(selected_elements).squeeze()
    return output

def resolve_sign(Tensor, resolve_method):
    if resolve_method == "mass":
        sign_to_mult = torch.sign(Tensor.sum(dim=1))  # 各个向量 对应维度 的求和后 对应的符号： + - 0，(batch, head*dim)
    elif resolve_method == "normmass":
        sign_to_mult = normmass_based_sign(Tensor)
    elif resolve_method == "normfrac":
        sign_to_mult = normfrac_based_sign(Tensor)
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")

    return sign_to_mult  # (batch, head*dim)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum(dim=1))
    if method == "majority":
        for i in range(sign_to_mult.shape[0]):
            sign_to_mult[i][sign_to_mult[i] == 0] = majority_sign[i]
    elif method == "minority":
        for i in range(sign_to_mult.shape[0]):
            sign_to_mult[i][sign_to_mult[i] == 0] = -1 * majority_sign[i]
    return sign_to_mult


def disjoint_merge(Tensor, merge_func, sign_to_mult):  # sign_to_mult: batch, head*dim; Tensor: batch, n_neg, head*dim
    merge_func = merge_func.split("-")[-1]  # mean
    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(1) > 0, Tensor > 0, Tensor < 0
        )
        selected_entries = Tensor * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=1).float()
        disjoint_aggs = torch.sum(selected_entries, dim=1) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=1)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=1)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs  # batch, head*dim
