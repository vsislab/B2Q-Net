import torch
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_phase import PhaseModel
import os
import pandas as pd
from configs.utils import setup_cfg

opts = parser.parse_args()
cfg = setup_cfg(opts.cfg_file, opts.set_cfgs)

out_folder = os.path.dirname(os.path.dirname(opts.resume)).replace('/checkpoints/','/predictions/')
gt_folder = os.path.join(out_folder,'gt')
pred_folder = os.path.join(out_folder,'pred')
os.makedirs(gt_folder,exist_ok=True)
os.makedirs(pred_folder,exist_ok=True)

if opts.task == 'phase':
	model = PhaseModel(opts, cfg, train=False)

if opts.only_temporal:
	_,_,test_set = prepare_image_features(model.net,opts,test_mode=True)
else:
	_,_,test_set = prepare_dataset(opts)

with torch.no_grad():

	if opts.cheat:
		model.net.train()
	else:
		model.net.eval()

	for ID,op in test_set:

		predictions = []
		labels = []

		if not opts.image_based:
			if opts.head == 'B2Q-Net':
				model.net.temporal_head.reset()
				model.net.temporal_head.block_list[0].frame_branch.reset()
			else:
				model.net.temporal_head.reset()
		
		model.metric_meter['test'].start_new_op()

		for data,target in op:

			data, target = prepare_batch(data,target)

			if opts.shuffle:
				if opts.head == 'B2Q-Net':
					model.net.temporal_head.reset()
					model.net.temporal_head.block_list[0].frame_branch.reset()
				else:
					model.net.temporal_head.reset()

			if opts.sliding_window:
				output = model.forward_sliding_window(data)
			else:
				_, output = model.forward(data, target, compute_loss=False)
			
			model.update_stats(
				0,
				output,
				target,
				mode='test'
			)

			if opts.task == 'phase':
				_,pred = torch.from_numpy(output[-1]['pred']).to(target.device).max(dim=-1)
				predictions.append(pred.flatten())
				labels.append(target.flatten())

		predictions = torch.cat(predictions)
		labels = torch.cat(labels)
	
		if opts.task == 'phase':
			predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Phase'])
			labels = pd.DataFrame(labels.cpu().numpy(),columns=['Phase'])

		predictions.to_csv(os.path.join(pred_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		labels.to_csv(os.path.join(gt_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		print('saved predictions/labels for video {}'.format(ID))

	epoch = torch.load(opts.resume)['epoch']
	model.summary(epoch=epoch)

