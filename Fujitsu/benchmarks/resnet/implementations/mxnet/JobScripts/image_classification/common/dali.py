import warnings
from nvidia import dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.mxnet import DALIClassificationIterator
from mlperf_compliance import constants as mlperf_constants
from mlperf_compliance import tags as mlperf_log
from mlperf_log_utils import mx_resnet_print
from distutils.util import strtobool


def add_dali_args(parser):
    group = parser.add_argument_group('DALI', 'pipeline and augumentation')
    group.add_argument('--data-train', type=str, help='the training data')
    group.add_argument('--data-train-idx', type=str, default='', help='the index of training data')
    group.add_argument('--data-val', type=str, help='the validation data')
    group.add_argument('--data-val-idx', type=str, default='', help='the index of validation data')
    group.add_argument('--use-dali', action='store_true',
                       help='use dalli pipeline and augunetation')
    group.add_argument('--max-random-area', type=float, default=1,
                       help='max area to crop in random resized crop, whose range is [0, 1]')
    group.add_argument('--min-random-area', type=float, default=1,
                       help='min area to crop in random resized crop, whose range is [0, 1]')
    group.add_argument('--separ-val', action='store_true',
                       help='each process will perform independent validation on whole val-set')
    group.add_argument('--min-random-aspect-ratio', type=float, default=3./4.,
                       help='min value of aspect ratio, whose value is either None or a positive value.')
    group.add_argument('--max-random-aspect-ratio', type=float, default=4./3.,
                       help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                            'the aspect ratio range is [1-max_random_aspect_ratio, '
                            '1+max_random_aspect_ratio], otherwise it is '
                            '[min_random_aspect_ratio, max_random_aspect_ratio].')
    group.add_argument('--dali-threads', type=int, default=3, help="number of threads" +\
                       "per GPU for DALI")
    group.add_argument('--image-shape', type=str,
                       help='the image shape feed into the network, e.g. (3,224,224)')
    group.add_argument('--num-examples', type=int, help='the number of training examples')
    
    group.add_argument('--dali-prefetch-queue', type=int, default=3, help="DALI prefetch queue depth")
    group.add_argument('--dali-nvjpeg-memory-padding', type=int, default=16, help="Memory padding value for nvJPEG (in MB)")
    group.add_argument('--max-random-area-2', type=float, default=1,
                       help='max area to crop in random resized crop, whose range is [0, 1]')
    group.add_argument('--min-random-area-2', type=float, default=1,
                       help='min area to crop in random resized crop, whose range is [0, 1]')
    group.add_argument('--min-random-aspect-ratio-2', type=float, default=3./4.,
                       help='min value of aspect ratio, whose value is either None or a positive value.')
    group.add_argument('--max-random-aspect-ratio-2', type=float, default=4./3.,
                       help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                            'the aspect ratio range is [1-max_random_aspect_ratio, '
                            '1+max_random_aspect_ratio], otherwise it is '
                            '[min_random_aspect_ratio, max_random_aspect_ratio].')
    group.add_argument('--use-new-cval', type=strtobool, default=False,
                       help='use new cval pipe like train pipe')

    return parser


_mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
_std_pixel  = [255 * x for x in (0.229, 0.224, 0.225)]

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, 
                 min_random_area, max_random_area,
                 min_random_aspect_ratio, max_random_aspect_ratio,
                 nvjpeg_padding, prefetch_queue=3,
                 seed=12,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True):
        super(HybridTrainPipe, self).__init__(
                batch_size, num_threads, device_id, 
                seed = seed + device_id, 
                prefetch_queue_depth = prefetch_queue)

        if mlperf_print:
            # Shuffiling is done inside ops.MXNetReader
            mx_resnet_print(key=mlperf_log.INPUT_ORDER)

        self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                     random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                        device_memory_padding = nvjpeg_padding,
                                        host_memory_padding = nvjpeg_padding)

        self.rrc = ops.RandomResizedCrop(device = "gpu",
                                         random_area = [
                                             min_random_area,
                                             max_random_area],
                                         random_aspect_ratio = [
                                             min_random_aspect_ratio,
                                             max_random_aspect_ratio],
                                         size = crop_shape)

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)
        self.coin = ops.CoinFlip(probability = 0.5)

        if mlperf_print:
            mx_resnet_print(
                    key=mlperf_log.INPUT_CROP_USES_BBOXES,
                    val=False)
            mx_resnet_print(
                    key=mlperf_log.INPUT_DISTORTED_CROP_RATIO_RANGE,
                    val=(min_random_aspect_ratio,
                         max_random_aspect_ratio))
            mx_resnet_print(
                    key=mlperf_log.INPUT_DISTORTED_CROP_AREA_RANGE,
                    val=(min_random_area,
                         max_random_area))
            mx_resnet_print(
                    key=mlperf_log.INPUT_MEAN_SUBTRACTION,
                    val=_mean_pixel)
            mx_resnet_print(
                    key=mlperf_log.INPUT_RANDOM_FLIP)


    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror = rng)
        return [output, self.labels]

class HybridCvalPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, 
                 nvjpeg_padding, prefetch_queue=3,
                 seed=12, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True):
        super(HybridCvalPipe, self).__init__(
                batch_size, num_threads, device_id, 
                seed = seed + device_id, 
                prefetch_queue_depth = prefetch_queue)

        self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                        device_memory_padding = nvjpeg_padding,
                                        host_memory_padding = nvjpeg_padding)

        self.resize = ops.Resize(device = "gpu", resize_shorter=resize_shp) if resize_shp else None

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)

        if mlperf_print:
            mx_resnet_print(
                    key=mlperf_log.INPUT_MEAN_SUBTRACTION,
                    val=_mean_pixel)
            mx_resnet_print(
                    key=mlperf_log.INPUT_RESIZE_ASPECT_PRESERVING)
            mx_resnet_print(
                    key=mlperf_log.INPUT_CENTRAL_CROP)


    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                 shard_id, num_shards, crop_shape, 
                 nvjpeg_padding, prefetch_queue=3,
                 seed=12, resize_shp=None,
                 output_layout=types.NCHW, pad_output=True, dtype='float16',
                 mlperf_print=True):

        super(HybridValPipe, self).__init__(
                batch_size, num_threads, device_id, 
                seed = seed + device_id,
                prefetch_queue_depth = prefetch_queue)

        self.input = ops.MXNetReader(path = [rec_path], index_path=[idx_path],
                                     random_shuffle=False, shard_id=shard_id, num_shards=num_shards)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB,
                                        device_memory_padding = nvjpeg_padding,
                                        host_memory_padding = nvjpeg_padding)

        self.resize = ops.Resize(device = "gpu", resize_shorter=resize_shp) if resize_shp else None

        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            output_dtype = types.FLOAT16 if dtype == 'float16' else types.FLOAT,
                                            output_layout = output_layout,
                                            crop = crop_shape,
                                            pad_output = pad_output,
                                            image_type = types.RGB,
                                            mean = _mean_pixel,
                                            std =  _std_pixel)

        if mlperf_print:
            mx_resnet_print(
                    key=mlperf_log.INPUT_MEAN_SUBTRACTION,
                    val=_mean_pixel)
            mx_resnet_print(
                    key=mlperf_log.INPUT_RESIZE_ASPECT_PRESERVING)
            mx_resnet_print(
                    key=mlperf_log.INPUT_CENTRAL_CROP)


    def define_graph(self):
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        if self.resize:
            images = self.resize(images)
        output = self.cmnp(images)
        return [output, self.labels]


def build_input_pipeline(args, kv=None):
    # resize is default base length of shorter edge for dataset;
    # all images will be reshaped to this size
    resize = int(args.resize)
    # target shape is final shape of images pipelined to network;
    # all images will be cropped to this size
    target_shape = tuple([int(l) for l in args.image_shape.split(',')])

    pad_output = target_shape[0] == 4
    gpus = list(map(int, filter(None, args.gpus.split(',')))) # filter to not encount eventually empty strings
    batch_size = args.batch_size//len(gpus)
    
    mx_resnet_print(
            key=mlperf_constants.MODEL_BN_SPAN,
            val=batch_size)
    
    num_threads = args.dali_threads

    # the input_layout w.r.t. the model is the output_layout of the image pipeline
    output_layout = types.NHWC if args.input_layout == 'NHWC' else types.NCHW

    rank = kv.rank if kv else 0
    nWrk = kv.num_workers if kv else 1


    trainpipes = [HybridTrainPipe(batch_size      = batch_size,
                                  num_threads     = num_threads,
                                  device_id       = gpu_id,
                                  rec_path        = args.data_train,
                                  idx_path        = args.data_train_idx,
                                  shard_id        = gpus.index(gpu_id) + len(gpus)*rank,
                                  num_shards      = len(gpus)*nWrk,
                                  crop_shape      = target_shape[1:],
                                  min_random_area = args.min_random_area,
                                  max_random_area = args.max_random_area,
                                  min_random_aspect_ratio = args.min_random_aspect_ratio,
                                  max_random_aspect_ratio = args.max_random_aspect_ratio,
                                  nvjpeg_padding  = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                  prefetch_queue  = args.dali_prefetch_queue,
                                  seed            = args.seed,
                                  output_layout   = output_layout,
                                  pad_output      = pad_output,
                                  dtype           = args.dtype,
                                  mlperf_print    = gpu_id == gpus[0]) for gpu_id in gpus]

    if args.no_augument_epoch >= args.num_epochs:
        cvalpipes = None
    else:
        cvalpipes = [HybridTrainPipe(batch_size      = batch_size,
                                     num_threads     = num_threads,
                                     device_id       = gpu_id,
                                     rec_path        = args.data_train,
                                     idx_path        = args.data_train_idx,
                                     shard_id        = gpus.index(gpu_id) + len(gpus)*rank,
                                     num_shards      = len(gpus)*nWrk,
                                     crop_shape      = target_shape[1:],
                                     min_random_area = args.min_random_area_2,
                                     max_random_area = args.max_random_area_2,
                                     min_random_aspect_ratio = args.min_random_aspect_ratio_2,
                                     max_random_aspect_ratio = args.max_random_aspect_ratio_2,
                                     nvjpeg_padding  = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                     prefetch_queue  = args.dali_prefetch_queue,
                                     seed            = args.seed,
                                     output_layout   = output_layout,
                                     pad_output      = pad_output,
                                     dtype           = args.dtype,
                                     mlperf_print    = gpu_id == gpus[0]) for gpu_id in gpus] if args.use_new_cval else [
                  HybridCvalPipe(batch_size      = batch_size,
                                    num_threads     = num_threads,
                                    device_id       = gpu_id,
                                    rec_path        = args.data_train,
                                    idx_path        = args.data_train_idx,
                                    shard_id        = gpus.index(gpu_id) + len(gpus)*rank,
                                    num_shards      = len(gpus)*nWrk,
                                    crop_shape      = target_shape[1:],
                                    nvjpeg_padding  = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                                    prefetch_queue  = args.dali_prefetch_queue,
                                    seed            = args.seed,
                                    resize_shp      = resize,
                                    output_layout   = output_layout,
                                    pad_output      = pad_output,
                                    dtype           = args.dtype,
                                    mlperf_print    = gpu_id == gpus[0]) for gpu_id in gpus]

    valpipes = [HybridValPipe(batch_size     = batch_size,
                              num_threads    = num_threads,
                              device_id      = gpu_id,
                              rec_path       = args.data_val,
                              idx_path       = args.data_val_idx,
                              shard_id       = 0 if args.separ_val
                                                 else gpus.index(gpu_id) + len(gpus)*rank,
                              num_shards     = 1 if args.separ_val else len(gpus)*nWrk,
                              crop_shape     = target_shape[1:],
                              nvjpeg_padding = args.dali_nvjpeg_memory_padding * 1024 * 1024,
                              prefetch_queue = args.dali_prefetch_queue,
                              seed           = args.seed,
                              resize_shp     = resize,
                              output_layout  = output_layout,
                              pad_output     = pad_output,
                              dtype          = args.dtype,
                              mlperf_print   = gpu_id == gpus[0]) for gpu_id in gpus] if args.data_val else None
    
    trainpipes[0].build()
    if args.no_augument_epoch < args.num_epochs:
        cvalpipes[0].build()
    if args.data_val:
        valpipes[0].build()

    return lambda args, kv: get_rec_iter(args, trainpipes, valpipes, cvalpipes, kv)


def get_rec_iter(args, trainpipes, valpipes, cvalpipes, kv=None):
    rank = kv.rank if kv else 0
    nWrk = kv.num_workers if kv else 1

    dali_train_iter = DALIClassificationIterator(trainpipes, args.num_examples // nWrk)
    if args.no_augument_epoch < args.num_epochs:
        dali_cval_iter = DALIClassificationIterator(cvalpipes, args.num_examples // nWrk)
    else:
        dali_cval_iter = None

    mx_resnet_print(
            key=mlperf_log.INPUT_SIZE,
            val=trainpipes[0].epoch_size("Reader"))

    mx_resnet_print(
            key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
            val=trainpipes[0].epoch_size("Reader"))

    if args.data_val:
        mx_resnet_print(
                key=mlperf_log.EVAL_SIZE,
                val=valpipes[0].epoch_size("Reader"))

        mx_resnet_print(
                key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
                val=valpipes[0].epoch_size("Reader"))


    if args.num_examples < trainpipes[0].epoch_size("Reader"):
        warnings.warn("{} training examples will be used, although full training set contains {} examples".format(args.num_examples, trainpipes[0].epoch_size("Reader")))

    worker_val_examples = valpipes[0].epoch_size("Reader")
    if not args.separ_val:
        worker_val_examples = worker_val_examples // nWrk
        if rank < valpipes[0].epoch_size("Reader") % nWrk:
            worker_val_examples += 1

    dali_val_iter = DALIClassificationIterator(valpipes, worker_val_examples, fill_last_batch = False) if args.data_val else None
    return dali_train_iter, dali_val_iter, dali_cval_iter

