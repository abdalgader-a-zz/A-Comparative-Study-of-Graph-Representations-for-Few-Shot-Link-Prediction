import torch
from collections import OrderedDict
from utils.utils import  val, test, EarlyStopping
from torchviz import make_dot
import wandb
from copy import deepcopy
import time

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def fine_tune_method  (model,
                       args,
                       data_batch,
                       optimiser,
                       inner_train_steps,
                       graph_id,
                       mode,
                       inner_avg_auc_list,
                       inner_avg_ap_list,
                       batch_id,
                       train,
                       inner_test_auc_array=None,
                       inner_test_ap_array=None):
    """
    Perform a gradient step on a meta-learner.
    # Arguments
        model: Base model of the meta-learner being trained
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        data_batch: Input samples for all few shot tasks
            meta-gradients after applying the update to
        inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
        graph_id: The ID of graph currently being trained
        train: Whether to update the meta-learner weights at the end of the episode.
        inner_test_auc_array: Final Test AUC array where we train to convergence
        inner_test_ap_array: Final Test AP array where we train to convergence
    """

    task_losses = []
    auc_list = []
    ap_list = []
    torch.autograd.set_detect_anomaly(True)


    for idx, data_graph in enumerate(data_batch):

        data_graph.train_mask = data_graph.val_mask = data_graph.test_mask = data_graph.y = None
        data_graph.batch = None
        num_nodes = data_graph.num_nodes

        if args.use_fixed_feats:
            perm = torch.randperm(args.feats.size(0))
            perm_idx = perm[:num_nodes]
            data_graph.x = args.feats[perm_idx]
        elif args.use_same_fixed_feats:
            node_feats = args.feats[0].unsqueeze(0).repeat(num_nodes,1)
            data_graph.x = node_feats

        if args.concat_fixed_feats:
            if data_graph.x.shape[1] < args.num_features:
                concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                data_graph.x = torch.cat((data_graph.x,concat_feats),1)

        # Val Ratio is Fixed at 0.1
        meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio

        ''' Check if Data is split'''
        try:
            x, train_pos_edge_index = data_graph.x.to(args.dev), data_graph.train_pos_edge_index.to(args.dev)
            data = data_graph
        except:
            data_graph.x.to(args.dev)
            data = model.split_edges(data_graph,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)

        # Additional Failure Checks for small graphs
        if data.val_pos_edge_index.size()[1] == 0 or data.test_pos_edge_index.size()[1] == 0:
            args.fail_counter += 1
            print("Failed on Graph %d" %(graph_id))
            continue

        try:
            x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
            test_pos_edge_index, test_neg_edge_index = data.test_pos_edge_index.to(args.dev),\
                    data.test_neg_edge_index.to(args.dev)
        except:
            print("Failed Splitting data on Graph %d" %(graph_id))
            continue

        early_stopping = EarlyStopping(patience=args.patience, verbose=False)
        start_time = time.time()

        if args.encoder == 'DGCNN':
            x = x.unsqueeze(0).permute(0, 2, 1)

        for inner_batch in range(inner_train_steps):
            model.train()
            optimiser.zero_grad()

            # Perform update of model weights
            if args.encoder == 'DGCNN':
                z = model.encode(x, OrderedDict(model.named_parameters()))
                z = z.squeeze(0)
            else:
                z = model.encode(x, train_pos_edge_index, OrderedDict(model.named_parameters()),\
                                 only_gae=args.apply_gae_only, inner_loop=True, train=train, no_sig=args.no_sig)

            loss = model.recon_loss(z, train_pos_edge_index)
            if args.model in ['VGAE']:
                if not args.apply_gae_only:
                    kl_loss = args.kl_anneal*(1 / num_nodes) * model.kl_loss()
                    loss = loss + kl_loss


            loss.backward()
            optimiser.step()

            ''' Only do this if its the final test set eval '''
            if args.final_test and inner_batch % 5 ==0:

                inner_test_auc, inner_test_ap = test(args, model, x, train_pos_edge_index, args.apply_gae_only,
                        data.test_pos_edge_index, data.test_neg_edge_index,OrderedDict(model.named_parameters()))
                val_pos_edge_index = data.val_pos_edge_index.to(args.dev)
                val_loss = val(model,args, x, args.apply_gae_only,val_pos_edge_index,data.num_nodes,OrderedDict(model.named_parameters()))
                early_stopping(val_loss, model, args, final=True)
                my_step = int(inner_batch / 5)
                inner_test_auc_array[graph_id][my_step] = inner_test_auc
                inner_test_ap_array[graph_id][my_step] = inner_test_ap


        # Do a pass of the model on the validation data from the current task
        val_pos_edge_index = data.val_pos_edge_index.to(args.dev)

        if args.encoder == 'DGCNN':
            z_val = model.encode(x, OrderedDict(model.named_parameters()))
            z_val = z_val.squeeze(0)
        else:
            z_val = model.encode(x, val_pos_edge_index, OrderedDict(model.named_parameters()),\
                                 only_gae=args.apply_gae_only, inner_loop=False, train=train, no_sig=args.no_sig)

        loss_val = model.recon_loss(z_val, val_pos_edge_index)
        if args.model in ['VGAE']:
            if not args.apply_gae_only:
                kl_loss = args.kl_anneal*(1 / num_nodes) * model.kl_loss()
                loss_val = loss_val + kl_loss

        if args.wandb:
            if args.model in ['VGAE']:
                wandb.log({f"Inner_Val_Total-loss of {mode} Graph {graph_id}":loss_val.item()})
                wandb.log({f"Inner_Val_Recon-loss of {mode} Graph {graph_id}":(loss_val.item()-kl_loss.item())})
                wandb.log({f"Inner_Val_Kl-loss of {mode} Graph {graph_id}":kl_loss.item()})
            else:
                wandb.log({f"Inner_Val_loss of {mode} Graph {graph_id}":loss_val.item()})

        # Get post-update accuracies
        auc, ap = test(args, model, x, train_pos_edge_index, args.apply_gae_only,
                data.test_pos_edge_index, data.test_neg_edge_index,OrderedDict(model.named_parameters()))

        auc_list.append(auc)
        ap_list.append(ap)
        inner_avg_auc_list.append(auc)
        inner_avg_ap_list.append(ap)

        # Accumulate losses and gradients
        graph_id += 1
        task_losses.append(loss_val)

    if args.no_meta_update:
        print('Inner Graph Batch: {:01d}, Inner-Update AUC: {:.4f}, AP: {:.4f} --- ({:.4f} minutes)'.format(batch_id,\
                            sum(auc_list) / len(auc_list), sum(ap_list) / len(ap_list), (time.time()-start_time)/60))

    # if args.wandb:
    #     if len(ap_list) > 0:
    #         auc_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AUC'
    #         ap_metric = mode + '_Local_Batch_Graph_' + str(batch_id) + '_AP'
    #         avg_auc_metric = mode + '_Inner_Batch_Graph' + '_AUC'
    #         avg_ap_metric = mode + '_Inner_Batch_Graph' + '_AP'
    #         wandb.log({auc_metric:sum(auc_list)/len(auc_list),ap_metric:sum(ap_list)/len(ap_list),\
    #                 avg_auc_metric:sum(auc_list)/len(auc_list),avg_ap_metric:sum(ap_list)/len(ap_list)})


    if len(task_losses) != 0:
        meta_batch_loss = torch.stack(task_losses).mean()
        return graph_id, meta_batch_loss, inner_avg_auc_list, inner_avg_ap_list



