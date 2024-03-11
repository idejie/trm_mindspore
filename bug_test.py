if __name__ == '__main__':
    from trm.config import cfg
    from trm.modeling import build_model
    import numpy as np
    from mindspore import Tensor,nn
    from mindspore import context
    from trm.data import make_data_loader
    import mindspore

    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    cfg.merge_from_file('configs/activitynet.yaml')
    cfg.freeze()        
    data_loader = make_data_loader(cfg, is_train=True)
    model = build_model(cfg)
    learning_rate = 1e-4
    bert_params = list(filter(lambda x: 'bert' in x.name, model.trainable_params()))
    base_params = list(filter(lambda x: 'bert' not in x.name, model.trainable_params()))

    # 基于多项式衰减函数计算学习率
    polynomial_decay_lr1 = nn.PolynomialDecayLR(learning_rate=learning_rate,      # 学习率初始值
                                            end_learning_rate=learning_rate*0.001, # 学习率最终值
                                            decay_steps=4,          # 衰减的step数
                                            power=0.5)              # 多项式幂
    
    polynomial_decay_lr2 = nn.PolynomialDecayLR(learning_rate=learning_rate*0.1,      # 学习率初始值
                                            end_learning_rate=learning_rate*0.0001, # 学习率最终值
                                            decay_steps=4,          # 衰减的step数
                                            power=0.5)              # 多项式幂
    
    group_params = [{'params': bert_params, 'weight_decay': 0.01, 'lr': polynomial_decay_lr2},
                {'params': base_params, 'lr': polynomial_decay_lr1}]
    
    optimizer = nn.AdamWeightDecay(group_params, learning_rate=learning_rate, weight_decay=1e-5)
    
    def forward_fn(model,batches,epoch=0):
        contrastive_scores, iou_scores, loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = model(batches,cur_epoch=epoch)
        print(f'forward finished')
        loss = loss_vid+loss_sent+loss_iou_stnc+loss_iou_phrase+scoremap_loss_pos+scoremap_loss_neg+scoremap_loss_exc
        print(f'loss: {loss} loss_vid: {loss_vid} loss_sent: {loss_sent} loss_iou_stnc: {loss_iou_stnc} loss_iou_phrase: {loss_iou_phrase} scoremap_loss_pos: {scoremap_loss_pos} scoremap_loss_neg: {scoremap_loss_neg} scoremap_loss_exc: {scoremap_loss_exc}')
        return loss,contrastive_scores, iou_scores
    
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    for iteration, batches in enumerate(data_loader.create_dict_iterator(num_epochs=1)):
        (loss, contrastive_scores, iou_scores), grads = grad_fn(model,batches)