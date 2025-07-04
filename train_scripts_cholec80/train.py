import torch
from random import shuffle
from tqdm import tqdm
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_phase import PhaseModel
import util_train as util
from configs.utils import cfg2flatdict, setup_cfg
import pickle
import os

opts = parser.parse_args()
cfg = setup_cfg(opts.cfg_file, opts.set_cfgs)

if not opts.random_seed:
	torch.manual_seed(7)

if cfg.Loss.nullw == -1:
    util.compute_null_weight(cfg)

if opts.task == 'phase':
	model = PhaseModel(opts, cfg)

if opts.only_temporal:
	train_set, val_set, test_set = prepare_image_features(model.net,opts)
else:
	train_set, val_set, test_set = prepare_dataset(opts)


with open(model.log_path, "w") as log_file:

	log_file.write(f'{model}\n')
	log_file.flush()

	start_epoch = util.get_start_epoch(opts)
	num_iters_per_epoch = util.get_iters_per_epoch(train_set,opts)
	
	target_dir = os.path.dirname(model.log_path) 
	pkl_path = os.path.join(target_dir, 'result.pkl')
	results = {
		'loss_train': [],
		'acc_train': [],
		'p_test': [],
		'r_test': [],
		'j_test': [],
		'f1_test': [],
		'ba_test': [],
		'acc_test': [],
	}

	for epoch in range(start_epoch,opts.epochs+1):

		model.reset_stats()
		model.net.train()
		if opts.bn_off:
			model.net.cnn.eval()

		for _,op in tqdm(train_set):

			if not opts.image_based:
				if opts.head == 'B2Q-Net':
					model.net.temporal_head.reset()
					model.net.temporal_head.block_list[0].frame_branch.reset()
				else:
					model.net.temporal_head.reset()

			model.metric_meter['train'].start_new_op() # necessary to compute video-wise metrics

			for i, (data, target) in enumerate(op):

				if not opts.image_based and opts.shuffle:
					if opts.head == 'B2Q-Net':
						model.net.temporal_head.reset()
						model.net.temporal_head.block_list[0].frame_branch.reset()
					else:
						model.net.temporal_head.reset()

				data, target = prepare_batch(data,target)
				
				loss, video_saves = model.forward(data, target, compute_loss=True)
				model.update_weights(loss)

				model.update_stats(
					loss.item(),
					video_saves,
					target,
					mode='train'
				)

				if opts.shuffle and (i+1) >= num_iters_per_epoch:
					break

				#break
			#break


		with torch.no_grad():

			if opts.cheat:
				model.net.train()
			else:
				model.net.eval()

			for mode in ['val','test']:

				if mode == 'val':
					eval_set = tqdm(val_set)
				elif mode == 'test':
					eval_set = tqdm(test_set)

				for _,op in eval_set:

					if not opts.image_based:
						if opts.head == 'B2Q-Net':
							model.net.temporal_head.reset()
							model.net.temporal_head.block_list[0].frame_branch.reset()
						else:
							model.net.temporal_head.reset()

					model.metric_meter[mode].start_new_op()

					for data,target in op:

						data, target = prepare_batch(data,target)

						if opts.sliding_window:
							output = model.forward_sliding_window(data)
							loss = model.compute_loss(output,target)
						else:
							loss, video_saves = model.forward(data, target, compute_loss=True)


						model.update_stats(
							loss.item(),
							video_saves,
							target,
							mode=mode
						)

						#break
					#break
					
			model.summary(log_file,epoch)
		
			loss_train, acc_train, _, _, _, _, _, _ = model.metric_meter['train'].get_scores()
			_, _, p_test, r_test, j_test, f1_test, ba_test, acc_test = model.metric_meter['test'].get_scores()

			results['loss_train'].append(loss_train)
			results['acc_train'].append(acc_train)
			results['p_test'].append(p_test)
			results['r_test'].append(r_test)
			results['j_test'].append(j_test)
			results['f1_test'].append(f1_test)
			results['ba_test'].append(ba_test)
			results['acc_test'].append(acc_test)

	with open(pkl_path, "wb") as f:
		pickle.dump(results, f)

