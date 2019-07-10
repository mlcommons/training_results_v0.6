PYTHONPATH=.:/tmp/code_dir-mask_code_1558420462/staging/models/rough/transformer/data_generators/:/tmp/code_dir-mask_code_1558420462/staging/models/rough/:$PYTHONPATH python3 mask_rcnn_main.py --hparams=first_lr_drop_step=9750,second_lr_drop_step=13000,total_steps=14784,train_batch_size=128,iterations_per_loop=924,eval_batch_size=256,train_use_tpu_estimator=false,eval_use_tpu_estimator=false \
--mode=train_and_eval \
--model_dir=gs://mlsh_test/dev/assets/model_dir-mask_model_dir_1558420462 \
--num_cores=32 \
--resnet_checkpoint=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603 \
--tpu=TEST_TPU_1558420477.5 \
--training_file_pattern=gs://mlperf-euw4/benchmark_data/maskrcnn_coco/train-* \
--use_tpu=true \
--val_json_file=gs://mlperf-euw4/benchmark_data/maskrcnn_coco/raw-data/annotations/instances_val2017.json \
--validation_file_pattern=gs://mlperf-euw4/benchmark_data/maskrcnn_coco/val-*
