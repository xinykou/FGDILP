from functools import partial
from mmengine import Registry
import torch
import torch.nn.functional as F

InnerDetoxHook = Registry('innerdetox_hook')


@InnerDetoxHook.register_module()
class BaseInnerDetoxHook():
    def __init__(self, norm_exp=0, neg_sim_exp=0, renorm=False, vector_correction=False):
        self.mem = dict()
        self.hook_handles = dict()
        self.norm_exp = norm_exp
        self.neg_sim_exp = neg_sim_exp
        self.renorm = renorm
        self.vector_correction = vector_correction
        self.lamda = 1
        self.batch_ids = None
        self.module_nm_to_ids = dict()  # 用于判断生成第几层的编码

    def read_hook(self, module, input, output, module_name=None, prompt_end_indices=None, ids=None, module_names_specialized=None):
        half_bsz = output.shape[0] // 2
        # batch, n_head, seq_len, dim = output.shape
        # neg - pos
        if self.mem.get(module_name, None) is None:
            self.mem[module_name] = dict()

        # todo: --------调整交叉注意力后的 "向量" 方向--------
        # if self.vector_correction:
        #     tmp = torch.zeros([half_bsz, n_head, 1, dim]).to(output.device)
        #     for index in range(half_bsz):
        #         vector_pos = output[index, :, -1, :].detach().flatten()
        #         vector_neg = output[index+half_bsz, :, -1, :].detach().flatten()
        #         vector_pos_norm = vector_pos / torch.linalg.norm(vector_pos)
        #         vector_neg_norm = vector_neg / torch.linalg.norm(vector_neg)
        #         vector_general = vector_pos_norm + vector_neg_norm
        #         general_vector_norm = vector_general / torch.linalg.norm(vector_general)
        #         vector_neg_general = torch.dot(vector_neg, general_vector_norm) * general_vector_norm
        #         vector_neg_rectify = vector_neg - vector_neg_general
        #         vector_delta = self.lamda * vector_neg_rectify - vector_pos
        #         tmp[index, :, :, :] = vector_delta.reshape(1, n_head, 1, dim)
        #     self.mem[module_name]['delta'] = tmp
            # print('vector build !!!')
        # ------------------------------------------------
        # if self.vector_correction:
        #     vector_pos = output[:half_bsz, :, -1:, :].detach()
        #     vector_neg = output[half_bsz:, :, -1:, :].detach()
        #     vector_pos_norm = vector_pos / torch.norm(vector_pos, p=2, dim=-1, keepdim=True)
        #     vector_neg_norm = vector_neg / torch.norm(vector_neg, p=2, dim=-1, keepdim=True)
        #     vector_general = vector_pos_norm + vector_neg_norm
        #     general_vector_norm = vector_general / torch.norm(vector_general, p=2, dim=-1, keepdim=True)
        #     vector_neg_general_unit = torch.mul(vector_neg, general_vector_norm).sum(dim=-1, keepdim=True)
        #     vector_neg_general = vector_neg_general_unit * general_vector_norm
        #     vector_neg_rectify = vector_neg - vector_neg_general
        #     vector_delta = self.lamda * vector_neg_rectify - vector_pos
        #     self.mem[module_name]['delta'] = vector_delta
        #     # print('vector')
        # else:
        self.mem[module_name]['delta'] = (output[half_bsz:,:,-1:,:] - output[:half_bsz,:,-1:,:]).detach()  # batch, n_head, 1, dim

        if self.batch_ids != ids:
            for key in module_names_specialized:
                self.module_nm_to_ids[key] = None
            self.module_nm_to_ids[module_name] = 1
            self.batch_ids = ids

            h,n,d = output.shape[1:]
            neg_end_indices = prompt_end_indices[half_bsz:,None,None,None].expand(-1, h, 1, d).to(output.device)  # batch * n_head * 1 * dim
            # self.mem[module_name]['neg_end'] : 1, n_head, 1, dim
            # ca = output.cpu().numpy()
            self.mem[module_name]['neg_end'] = torch.gather(output[half_bsz:, :, :, :], dim=2, index=neg_end_indices).detach()
                # torch.gather(output[half_bsz:,:,:,:], dim=2, index=neg_end_indices)[:1,:,:,:].detach()  # 一个批次只取了第一个 end位置的编码作为‘neg_end’
        elif self.batch_ids == ids and self.module_nm_to_ids.get(module_name, None) is None:  # 从第二层开始
            self.module_nm_to_ids[module_name] = 1
            h, n, d = output.shape[1:]
            neg_end_indices = prompt_end_indices[half_bsz:, None, None, None].expand(-1, h, 1, d).to(output.device)
            self.mem[module_name]['neg_end'] = torch.gather(output[half_bsz:, :, :, :], dim=2, index=neg_end_indices).detach()
        # else:
        #     raise ValueError('prompt_end_indices is None')
        # print(f"计算结束: batch-{self.batch_ids} module_name-{module_name}")


    def write_hook(self, module, input, output, module_name=None):
        try:
            return self.modification_fn(output, self.mem[module_name], module_name)
        except:
            print()

    def register_hooks(self, model, hook):
        for n, m in model.named_modules():
            if self.module_match_fn(n):   # todo: 定义我们在哪些位置 执行减法 做差
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
            v_norm = v[:,:,-1:,:].norm(dim=(1,3), keepdim=True)   # batch, n_head, 1, dim

        delta = v_mem['delta']  # batch,n_head, 1, dim
        neg_end = v_mem['neg_end']   # 1, n_head, 1, dim

        norm_scale = 1
        if self.norm_exp > 0:
            norm_scale = (1 + delta.norm(dim=-1, keepdim=True)) ** self.norm_exp

        neg_sim_scale = 1
        if self.neg_sim_exp > 0:
            neg_sim = (neg_end * v[:,:,-1:,:]).sum(dim=-1, keepdim=True) / (neg_end.norm(dim=-1, keepdim=True) * v[:,:,-1:,:].norm(dim=-1, keepdim=True))
            neg_sim_scale = (1 + F.relu(neg_sim)) ** self.neg_sim_exp

        v[:,:,-1:,:] = v[:,:,-1:,:] - norm_scale * neg_sim_scale * delta

        if self.renorm:
            new_v_norm = v[:,:,-1:,:].norm(dim=(1,3), keepdim=True)
            v[:,:,-1:,:] = v[:,:,-1:,:] * (v_norm / new_v_norm)
        return v


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