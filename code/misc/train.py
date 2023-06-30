import time
from munch import Munch
import torch
import copy
from misc.metric import MetricManager
from tqdm.auto import tqdm

class Trainer(object):
    def __init__(self, model, metric_manager):
        self.model = model

        self.metric_manager = metric_manager
        self.best_epoch = None
        self.best_param = None
        self.best_metric = self.metric_manager.init_best_metric()

        self.patience = 0

    def train(self, save_manager, train_loader, val_loader, optimizer, writer=None):
        for epoch in range(1, save_manager.config.Train.epoch+1):
            start_time = time.time()

            #train
            self.model.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                batch.detecting = False
                batch.is_training = True
                out, batch = self.forward_batch(batch)
                optimizer.update_model(out.loss)
                train_loss += out.loss.item()

                if writer is not None:
                    writer.write_loss(out.loss.item(), 'train')
                if i % 50 == 0:
                    save_manager.write_log('iter [{}/{}] loss [{:.6f}]'.format(i, len(train_loader), out.loss.item()))

            train_loss = train_loss / len(train_loader)
            train_metric = self.metric_manager.average_running_metric()

            if writer is not None:
                writer.write_metric(train_metric, 'train')

            #val
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for i, batch in enumerate(val_loader):
                    batch.detecting = False
                    batch.is_training = True
                    out, batch = self.forward_batch(batch)
                    val_loss += out.loss.item()

                    if writer is not None:
                        writer.write_loss(out.loss.item(), 'val')
                        if epoch % 20 == 0 and i == 0:
                            writer.plot_image_heatmap(image=batch.input_image.cpu(), pred_heatmap=out.pred.heatmap.cpu(), label_heatmap=batch.label.heatmap.cpu(), epoch=epoch)
                val_metric = self.metric_manager.average_running_metric()
                val_loss = val_loss / len(val_loader)

                optimizer.scheduler_step(val_metric[save_manager.config.Train.decision_metric])


                if writer is not None:
                    writer.write_metric(val_metric, 'val')
                    if epoch % 5 == 0:
                        writer.plot_model_param_histogram(self.model, epoch)

            save_manager.write_log('Epoch [{}/{}] train loss [{:.6f}] train {} [{:.6f}] val loss [{:.6f}] val {} [{:.6f}] Epoch time [{:.1f}min]'.format(
                                    epoch,
                                    save_manager.config.Train.epoch,
                                    train_loss,
                                    save_manager.config.Train.decision_metric,
                                    train_metric[save_manager.config.Train.decision_metric],
                                    val_loss,
                                    save_manager.config.Train.decision_metric,
                                    val_metric[save_manager.config.Train.decision_metric],
                                    (time.time()-start_time)/60))
            print('version: {}'.format(save_manager.config.version))
            if self.metric_manager.is_new_best(old=self.best_metric, new=val_metric):
                self.patience = 0
                self.best_epoch = epoch
                self.best_metric = val_metric
                save_manager.save_model(self.best_epoch, self.model.state_dict(), self.best_metric)
                save_manager.write_log('Model saved after Epoch {}'.format(epoch), 4)
            else :
                self.patience += 1
                if self.patience > save_manager.config.Train.patience:
                    save_manager.write_log('Training Early Stopped after Epoch {}'.format(epoch), 16)
                    break

    def test(self, save_manager, test_loader, writer=None):
        # To reduce GPU usage, load params on cpu and upload model on gpu(params on gpu also need additional GPU memory)
        self.model.cpu()
        self.model.load_state_dict(self.best_param)
        self.model.to(save_manager.config.MISC.device)


        save_manager.write_log('Best model at epoch {} loaded'.format(self.best_epoch))
        self.model.eval()
        with torch.no_grad():
            post_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]
            manual_metric_managers = [MetricManager(save_manager) for _ in range(save_manager.config.Dataset.num_keypoint+1)]

            for i, batch in enumerate(tqdm(test_loader)):
                batch = Munch.fromDict(batch)
                batch.detecting = False
                batch.is_training = False
                max_hint = 10+1
                for n_hint in range(max_hint):
                    # 0~max hint

                    ## model forward
                    out, batch, post_processing_pred = self.forward_batch(batch, metric_flag=True, average_flag=False, metric_manager=post_metric_managers[n_hint], return_post_processing_pred=True)
                    if save_manager.config.Model.use_prev_heatmap:
                        batch.prev_heatmap = out.pred.heatmap.detach()

                    ## manual forward
                    if n_hint == 0:
                        manual_pred = copy.deepcopy(out.pred)
                        manual_pred.sargmax_coord = manual_pred.sargmax_coord
                        manual_pred.heatmap = manual_pred.heatmap
                        manual_hint_index = []
                    else:
                        # manual prediction update
                        for k in range(manual_hint_index.shape[0]):
                            manual_pred.sargmax_coord[k, manual_hint_index[k]] = batch.label.coord[k, manual_hint_index[k]].to(manual_pred.sargmax_coord.device)
                            manual_pred.heatmap[k, manual_hint_index[k]] = batch.label.heatmap[k, manual_hint_index[k]].to(manual_pred.heatmap.device)
                    manual_metric_managers[n_hint].measure_metric(manual_pred, batch.label, batch.pspace, metric_flag=True, average_flag=False)

                    # ============================= model hint =================================
                    worst_index = self.find_worst_pred_index(batch.hint.index, post_metric_managers, save_manager, n_hint)
                    # hint index update
                    if n_hint == 0:
                        batch.hint.index = worst_index # (batch, 1)
                    else:
                        batch.hint.index = torch.cat((batch.hint.index, worst_index.to(batch.hint.index.device)), dim=1) # ... (batch, max hint)

                    #
                    if save_manager.config.Model.use_prev_heatmap_only_for_hint_index:
                        new_prev_heatmap = torch.zeros_like(out.pred.heatmap)
                        for j in range(len(batch.hint.index)):
                            new_prev_heatmap[j, batch.hint.index[j]] = out.pred.heatmap[j, batch.hint.index[j]]
                        batch.prev_heatmap = new_prev_heatmap
                    # ================================= manual hint =========================
                    worst_index = self.find_worst_pred_index(manual_hint_index, manual_metric_managers, save_manager, n_hint)
                    # hint index update
                    if n_hint == 0:
                        manual_hint_index = worst_index  # (batch, 1)
                    else:
                        manual_hint_index = torch.cat((manual_hint_index, worst_index.to(manual_hint_index.device)), dim=1)  # ... (batch, max hint)

                    # save result
                    if save_manager.config.save_test_prediction:
                        save_manager.add_test_prediction_for_save(batch, post_processing_pred, manual_pred, n_hint, post_metric_managers[n_hint], manual_metric_managers[n_hint])

            post_metrics = [metric_manager.average_running_metric() for metric_manager in post_metric_managers]
            manual_metrics = [metric_manager.average_running_metric() for metric_manager in manual_metric_managers]

        # save metrics
        for t in range(min(max_hint, len(post_metrics))):
            save_manager.write_log('(model ) Hint {} ::: {}'.format(t, post_metrics[t]))
            save_manager.write_log('(manual) Hint {} ::: {}'.format(t, manual_metrics[t]))
        save_manager.save_metric(post_metrics, manual_metrics)

        if save_manager.config.save_test_prediction:
            save_manager.save_test_prediction()

    def forward_batch(self, batch, metric_flag=False, average_flag=True, metric_manager=None, return_post_processing_pred=False):
        out, batch = self.model(batch)
        with torch.no_grad():
            if metric_manager is None:
                self.metric_manager.measure_metric(out.pred, batch.label, batch.pspace, metric_flag, average_flag)
            else:
                #post processing
                post_processing_pred = copy.deepcopy(out.pred)
                post_processing_pred.sargmax_coord = post_processing_pred.sargmax_coord.detach()
                post_processing_pred.heatmap = post_processing_pred.heatmap.detach()
                for i in range(len(batch.hint.index)): #for 문이 batch에 대해서 도는중 i번째 item
                    if batch.hint.index[i] is not None:
                        post_processing_pred.sargmax_coord[i, batch.hint.index[i]] = batch.label.coord[i, batch.hint.index[i]].detach().to(post_processing_pred.sargmax_coord.device)
                        post_processing_pred.heatmap[i, batch.hint.index[i]] = batch.label.heatmap[i, batch.hint.index[i]].detach().to(post_processing_pred.heatmap.device)

                metric_manager.measure_metric(post_processing_pred, copy.deepcopy(batch.label), batch.pspace, metric_flag=True, average_flag=False)
        if return_post_processing_pred:
            return out, batch, post_processing_pred
        else:
            return out, batch

    def find_worst_pred_index(self, previous_hint_index, metric_managers, save_manager, n_hint):
        batch_metric_value = metric_managers[n_hint].running_metric[save_manager.config.Train.decision_metric][-1]
        if len(batch_metric_value.shape) == 3:
            batch_metric_value = batch_metric_value.mean(-1)  # (batch, 13)

        tmp_metric = batch_metric_value.clone().detach()
        if metric_managers[n_hint].minimize_metric:
            for j, idx in enumerate(previous_hint_index):
                if idx is not None:
                    tmp_metric[j, idx] = -1000
            worst_index = tmp_metric.argmax(-1, keepdim=True)
        else:
            for j, idx in enumerate(previous_hint_index):
                if idx is not None:
                    tmp_metric[j, idx] = 1000
            worst_index = tmp_metric.argmin(-1, keepdim=True)
        return worst_index

# (model ) Hint 0 ::: Munch({'hargmax_pixel_MAE': 8.223318, 'hargmax_pixel_RMSE': 12.326061550527811, 'hargmax_pixel_MRE': 13.614773, 'hargmax_mm_MAE': 35.209145, 'hargmax_mm_RMSE': 54.121264435350895, 'hargmax_mm_MRE': 59.146645, 'sargmax_pixel_MAE': 8.090493, 'sargmax_pixel_RMSE': 12.207376735284925, 'sargmax_pixel_MRE': 13.414206, 'sargmax_mm_MAE': 34.622585, 'sargmax_mm_RMSE': 53.59625303372741, 'sargmax_mm_MRE': 58.259727}) 
# (manual) Hint 0 ::: Munch({'hargmax_pixel_MAE': 8.223318, 'hargmax_pixel_RMSE': 12.326061550527811, 'hargmax_pixel_MRE': 13.614773, 'hargmax_mm_MAE': 35.209145, 'hargmax_mm_RMSE': 54.121264435350895, 'hargmax_mm_MRE': 59.146645, 'sargmax_pixel_MAE': 8.090493, 'sargmax_pixel_RMSE': 12.207376735284925, 'sargmax_pixel_MRE': 13.414206, 'sargmax_mm_MAE': 34.622585, 'sargmax_mm_RMSE': 53.59625303372741, 'sargmax_mm_MRE': 58.259727}) 
# (model ) Hint 1 ::: Munch({'hargmax_pixel_MAE': 5.3337617, 'hargmax_pixel_RMSE': 7.987432622350752, 'hargmax_pixel_MRE': 8.730467, 'hargmax_mm_MAE': 22.13916, 'hargmax_mm_RMSE': 33.92658822610974, 'hargmax_mm_MRE': 36.61432, 'sargmax_pixel_MAE': 5.179494, 'sargmax_pixel_RMSE': 7.837093041278422, 'sargmax_pixel_MRE': 8.49528, 'sargmax_mm_MAE': 21.458916, 'sargmax_mm_RMSE': 33.25970681011677, 'sargmax_mm_MRE': 35.57628}) 
# (manual) Hint 1 ::: Munch({'hargmax_pixel_MAE': 7.8610787, 'hargmax_pixel_RMSE': 11.634128242731094, 'hargmax_pixel_MRE': 12.9880295, 'hargmax_mm_MAE': 33.626434, 'hargmax_mm_RMSE': 51.04874621331692, 'hargmax_mm_MRE': 56.374023, 'sargmax_pixel_MAE': 7.7294283, 'sargmax_pixel_RMSE': 11.519068137742579, 'sargmax_pixel_MRE': 12.789525, 'sargmax_mm_MAE': 33.04615, 'sargmax_mm_RMSE': 50.54108652845025, 'sargmax_mm_MRE': 55.4974}) 
# (model ) Hint 2 ::: Munch({'hargmax_pixel_MAE': 4.301047, 'hargmax_pixel_RMSE': 6.252836335450411, 'hargmax_pixel_MRE': 7.0045953, 'hargmax_mm_MAE': 17.549507, 'hargmax_mm_RMSE': 26.0250699557364, 'hargmax_mm_MRE': 28.81007, 'sargmax_pixel_MAE': 4.134078, 'sargmax_pixel_RMSE': 6.084671256132424, 'sargmax_pixel_MRE': 6.7473426, 'sargmax_mm_MAE': 16.81583, 'sargmax_mm_RMSE': 25.279711939394474, 'sargmax_mm_MRE': 27.673111}) 
# (manual) Hint 2 ::: Munch({'hargmax_pixel_MAE': 7.5263944, 'hargmax_pixel_RMSE': 10.982785305939615, 'hargmax_pixel_MRE': 12.408114, 'hargmax_mm_MAE': 32.163906, 'hargmax_mm_RMSE': 48.12754795700312, 'hargmax_mm_MRE': 53.80376, 'sargmax_pixel_MAE': 7.3946533, 'sargmax_pixel_RMSE': 10.869884595274925, 'sargmax_pixel_MRE': 12.209459, 'sargmax_mm_MAE': 31.583696, 'sargmax_mm_RMSE': 47.62850884720683, 'sargmax_mm_MRE': 52.92701}) 
# (model ) Hint 3 ::: Munch({'hargmax_pixel_MAE': 3.770506, 'hargmax_pixel_RMSE': 5.476892528124154, 'hargmax_pixel_MRE': 6.108018, 'hargmax_mm_MAE': 15.234551, 'hargmax_mm_RMSE': 22.507501997053623, 'hargmax_mm_MRE': 24.830013, 'sargmax_pixel_MAE': 3.6001904, 'sargmax_pixel_RMSE': 5.305076270364225, 'sargmax_pixel_MRE': 5.846118, 'sargmax_mm_MAE': 14.490898, 'sargmax_mm_RMSE': 21.75205648317933, 'sargmax_mm_MRE': 23.68177}) 
# (manual) Hint 3 ::: Munch({'hargmax_pixel_MAE': 7.2090373, 'hargmax_pixel_RMSE': 10.290395909920335, 'hargmax_pixel_MRE': 11.855768, 'hargmax_mm_MAE': 30.769218, 'hargmax_mm_RMSE': 44.98908917233348, 'hargmax_mm_MRE': 51.345253, 'sargmax_pixel_MAE': 7.0774703, 'sargmax_pixel_RMSE': 10.180644168518484, 'sargmax_pixel_MRE': 11.657513, 'sargmax_mm_MAE': 30.190134, 'sargmax_mm_RMSE': 44.503655422478914, 'sargmax_mm_MRE': 50.471115}) 
# (model ) Hint 4 ::: Munch({'hargmax_pixel_MAE': 3.3601809, 'hargmax_pixel_RMSE': 4.942305498756468, 'hargmax_pixel_MRE': 5.4534206, 'hargmax_mm_MAE': 13.601244, 'hargmax_mm_RMSE': 20.266624316573143, 'hargmax_mm_MRE': 22.18702, 'sargmax_pixel_MAE': 3.1870332, 'sargmax_pixel_RMSE': 4.766958610154688, 'sargmax_pixel_MRE': 5.1880074, 'sargmax_mm_MAE': 12.849865, 'sargmax_mm_RMSE': 19.495058361440897, 'sargmax_mm_MRE': 21.030888}) 
# (manual) Hint 4 ::: Munch({'hargmax_pixel_MAE': 6.8969965, 'hargmax_pixel_RMSE': 9.403292910195887, 'hargmax_pixel_MRE': 11.319115, 'hargmax_mm_MAE': 29.398497, 'hargmax_mm_RMSE': 40.93077493458986, 'hargmax_mm_MRE': 48.9573, 'sargmax_pixel_MAE': 6.7645273, 'sargmax_pixel_RMSE': 9.292931352742016, 'sargmax_pixel_MRE': 11.119673, 'sargmax_mm_MAE': 28.815739, 'sargmax_mm_RMSE': 40.44061504676938, 'sargmax_mm_MRE': 48.07834}) 
# (model ) Hint 5 ::: Munch({'hargmax_pixel_MAE': 3.0154533, 'hargmax_pixel_RMSE': 4.356896690092981, 'hargmax_pixel_MRE': 4.8748603, 'hargmax_mm_MAE': 12.132032, 'hargmax_mm_RMSE': 17.755203425884247, 'hargmax_mm_MRE': 19.68195, 'sargmax_pixel_MAE': 2.8322368, 'sargmax_pixel_RMSE': 4.166692851111293, 'sargmax_pixel_MRE': 4.5926685, 'sargmax_mm_MAE': 11.339841, 'sargmax_mm_RMSE': 16.92587736621499, 'sargmax_mm_MRE': 18.45607}) 
# (manual) Hint 5 ::: Munch({'hargmax_pixel_MAE': 6.6863766, 'hargmax_pixel_RMSE': 9.145693109370768, 'hargmax_pixel_MRE': 10.96677, 'hargmax_mm_MAE': 28.484385, 'hargmax_mm_RMSE': 39.78961869701743, 'hargmax_mm_MRE': 47.4097, 'sargmax_pixel_MAE': 6.5545125, 'sargmax_pixel_RMSE': 9.040146904066205, 'sargmax_pixel_MRE': 10.768182, 'sargmax_mm_MAE': 27.905266, 'sargmax_mm_RMSE': 39.32173718512058, 'sargmax_mm_MRE': 46.53563}) 
# (model ) Hint 6 ::: Munch({'hargmax_pixel_MAE': 2.8394394, 'hargmax_pixel_RMSE': 4.074789620004594, 'hargmax_pixel_MRE': 4.580435, 'hargmax_mm_MAE': 11.366426, 'hargmax_mm_RMSE': 16.540122915059328, 'hargmax_mm_MRE': 18.38286, 'sargmax_pixel_MAE': 2.6539793, 'sargmax_pixel_RMSE': 3.885408208705485, 'sargmax_pixel_MRE': 4.2944665, 'sargmax_mm_MAE': 10.563958, 'sargmax_mm_RMSE': 15.71198670938611, 'sargmax_mm_MRE': 17.139866}) 
# (manual) Hint 6 ::: Munch({'hargmax_pixel_MAE': 6.477401, 'hargmax_pixel_RMSE': 8.894469580613077, 'hargmax_pixel_MRE': 10.623891, 'hargmax_mm_MAE': 27.581001, 'hargmax_mm_RMSE': 38.68115724623203, 'hargmax_mm_MRE': 45.907234, 'sargmax_pixel_MAE': 6.3446846, 'sargmax_pixel_RMSE': 8.789277110248804, 'sargmax_pixel_MRE': 10.423939, 'sargmax_mm_MAE': 26.99836, 'sargmax_mm_RMSE': 38.214900966733694, 'sargmax_mm_MRE': 45.02767}) 
# (model ) Hint 7 ::: Munch({'hargmax_pixel_MAE': 2.6281505, 'hargmax_pixel_RMSE': 3.5597989475354552, 'hargmax_pixel_MRE': 4.219629, 'hargmax_mm_MAE': 10.472047, 'hargmax_mm_RMSE': 14.314435448497534, 'hargmax_mm_MRE': 16.843739, 'sargmax_pixel_MAE': 2.4418807, 'sargmax_pixel_RMSE': 3.3652511658146977, 'sargmax_pixel_MRE': 3.9315064, 'sargmax_mm_MAE': 9.670014, 'sargmax_mm_RMSE': 13.46866549178958, 'sargmax_mm_MRE': 15.600137}) 
# (manual) Hint 7 ::: Munch({'hargmax_pixel_MAE': 6.274641, 'hargmax_pixel_RMSE': 8.657198541797698, 'hargmax_pixel_MRE': 10.293919, 'hargmax_mm_MAE': 26.706541, 'hargmax_mm_RMSE': 37.63456832617521, 'hargmax_mm_MRE': 44.462715, 'sargmax_pixel_MAE': 6.1423683, 'sargmax_pixel_RMSE': 8.555035887286067, 'sargmax_pixel_MRE': 10.0938, 'sargmax_mm_MAE': 26.126514, 'sargmax_mm_RMSE': 37.181627467274666, 'sargmax_mm_MRE': 43.583252}) 
# (model ) Hint 8 ::: Munch({'hargmax_pixel_MAE': 2.5085342, 'hargmax_pixel_RMSE': 3.6652595810592175, 'hargmax_pixel_MRE': 4.041853, 'hargmax_mm_MAE': 9.958298, 'hargmax_mm_RMSE': 14.6176712885499, 'hargmax_mm_MRE': 16.048054, 'sargmax_pixel_MAE': 2.3221335, 'sargmax_pixel_RMSE': 3.47312390524894, 'sargmax_pixel_MRE': 3.7540772, 'sargmax_mm_MAE': 9.153828, 'sargmax_mm_RMSE': 13.780673246830702, 'sargmax_mm_MRE': 14.8011675}) 
# (manual) Hint 8 ::: Munch({'hargmax_pixel_MAE': 6.082141, 'hargmax_pixel_RMSE': 8.43211053404957, 'hargmax_pixel_MRE': 9.97609, 'hargmax_mm_MAE': 25.871422, 'hargmax_mm_RMSE': 36.63299546018243, 'hargmax_mm_MRE': 43.064415, 'sargmax_pixel_MAE': 5.9490895, 'sargmax_pixel_RMSE': 8.331205263733864, 'sargmax_pixel_MRE': 9.774797, 'sargmax_mm_MAE': 25.288769, 'sargmax_mm_RMSE': 36.18536011129618, 'sargmax_mm_MRE': 42.180748}) 
# (model ) Hint 9 ::: Munch({'hargmax_pixel_MAE': 2.308868, 'hargmax_pixel_RMSE': 3.162292491644621, 'hargmax_pixel_MRE': 3.7089016, 'hargmax_mm_MAE': 9.100821, 'hargmax_mm_RMSE': 12.522929646074772, 'hargmax_mm_MRE': 14.616888, 'sargmax_pixel_MAE': 2.125949, 'sargmax_pixel_RMSE': 2.974829907529056, 'sargmax_pixel_MRE': 3.4253678, 'sargmax_mm_MAE': 8.306095, 'sargmax_mm_RMSE': 11.697073198854923, 'sargmax_mm_MRE': 13.381232}) 
# (manual) Hint 9 ::: Munch({'hargmax_pixel_MAE': 5.894614, 'hargmax_pixel_RMSE': 8.21336582209915, 'hargmax_pixel_MRE': 9.667039, 'hargmax_mm_MAE': 25.062614, 'hargmax_mm_RMSE': 35.66174789890647, 'hargmax_mm_MRE': 41.70523, 'sargmax_pixel_MAE': 5.761061, 'sargmax_pixel_RMSE': 8.114407889544964, 'sargmax_pixel_MRE': 9.465132, 'sargmax_mm_MAE': 24.478844, 'sargmax_mm_RMSE': 35.223533038049936, 'sargmax_mm_MRE': 40.820396}) 
# (model ) Hint 10 ::: Munch({'hargmax_pixel_MAE': 2.1898618, 'hargmax_pixel_RMSE': 2.9899192498996854, 'hargmax_pixel_MRE': 3.5051532, 'hargmax_mm_MAE': 8.584366, 'hargmax_mm_RMSE': 11.788533110171556, 'hargmax_mm_MRE': 13.717706, 'sargmax_pixel_MAE': 2.0020075, 'sargmax_pixel_RMSE': 2.799793941900134, 'sargmax_pixel_MRE': 3.2151258, 'sargmax_mm_MAE': 7.769408, 'sargmax_mm_RMSE': 10.9564169049263, 'sargmax_mm_MRE': 12.456832}) 
# (manual) Hint 10 ::: Munch({'hargmax_pixel_MAE': 5.7117577, 'hargmax_pixel_RMSE': 8.000761703588068, 'hargmax_pixel_MRE': 9.365792, 'hargmax_mm_MAE': 24.271706, 'hargmax_mm_RMSE': 34.72215688228607, 'hargmax_mm_MRE': 40.38587, 'sargmax_pixel_MAE': 5.578359, 'sargmax_pixel_RMSE': 7.90409514401108, 'sargmax_pixel_MRE': 9.162828, 'sargmax_mm_MAE': 23.689365, 'sargmax_mm_RMSE': 34.29538822546601, 'sargmax_mm_MRE': 39.49817}) 
