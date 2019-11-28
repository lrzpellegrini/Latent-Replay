####################################################################################################
# Copyright (c) 2019. Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, Davide Maltoni    #
#                                                                                                  #
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file #
####################################################################################################

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


def dump(obj):
    for attr in dir(obj):
        if hasattr(obj, attr):
            # print( "obj.%s = %s" % (attr, getattr(obj, attr)))
            print("obj.%s" % (attr,))


def protobuf_insert_element_in_composite_container(index, element, container):
    container.add()
    for i in reversed(range(index+1, len(container))):
        container[i].CopyFrom(container[i-1])

    container[index].CopyFrom(element)

    # for i in range(index, index+3): # Debug
    #    print('At index', i, ':', container[i])
    # print('Last:', container[-1])
    return True


def protobuf_find_layer_by_name(net_param, layer_name, phase=None):
    index = 0
    for layer in net_param.layer:
        if layer.name == layer_name:
            if phase is None:
                return layer, index
            else:
                if layer.include[0].phase == phase:
                    return layer, index
        index += 1
    return None, -1


def create_concat_layer_from_net_template(orig_net, result_net, lat_layer_name, lat_layer_shape, stop_layer_name,
                                          original_input=21, rehearsal_input=107):
    net_param = caffe_pb2.NetParameter()
    with open(orig_net) as f:
        txtf.Merge(str(f.read()), net_param)
    data_layer, data_layer_index = protobuf_find_layer_by_name(net_param, 'data', caffe_pb2.TRAIN)
    if data_layer is None:
        raise RuntimeError('No data layer found (TRAIN phase)')

    data_layer.input_param.shape[0].dim[0] = original_input

    rehe_layer_param = caffe_pb2.LayerParameter()
    txtf.Merge('name: "data_reha"\n' +
               'type: "Input"\n' +
               'top: "data_reha"\n' +
               'include {\n' +
               '  phase: TRAIN\n' +
               '}\n' +
               'input_param {\n' +
               '  shape {\n' +
               '    dim: ' + str(rehearsal_input) + '\n' +
               '    dim: ' + str(lat_layer_shape[1]) + '\n' +
               '    dim: ' + str(lat_layer_shape[2]) + '\n' +
               '    dim: ' + str(lat_layer_shape[3]) + '\n' +
               '  }\n' +
               '}', rehe_layer_param)
    protobuf_insert_element_in_composite_container(data_layer_index + 1, rehe_layer_param, net_param.layer)

    lat_layer, lat_layer_index = protobuf_find_layer_by_name(net_param, lat_layer_name)
    if lat_layer is None:
        raise RuntimeError('No latent rehearsal layer found ' + lat_layer_name)

    concat_layer_param = caffe_pb2.LayerParameter()
    txtf.Merge('name: "concat"\n' +
               'type: "Concat"\n' +
               'bottom: "' + lat_layer_name + '"\n' +
               'bottom: "data_reha"\n' +
               'top: "concat"\n' +
               'include {\n' +
               '  phase: TRAIN\n' +
               '}\n' +
               'concat_param {\n' +
               '  axis: 0\n' +
               '}', concat_layer_param)

    stop_layer_train, stop_layer_train_index = protobuf_find_layer_by_name(net_param, stop_layer_name)
    if stop_layer_train is None:
        raise RuntimeError('No stop layer found ' + stop_layer_name)

    protobuf_insert_element_in_composite_container(stop_layer_train_index, concat_layer_param, net_param.layer)
    stop_layer_train, stop_layer_train_index = protobuf_find_layer_by_name(net_param, stop_layer_name)

    stop_layer_param_test = caffe_pb2.LayerParameter()
    stop_layer_param_test.CopyFrom(stop_layer_train)

    if len(stop_layer_train.include) == 0:
        stop_layer_train.include.add()
    stop_layer_train.include[0].phase = caffe_pb2.TRAIN
    stop_layer_train.bottom[0] = 'concat'

    if len(stop_layer_param_test.include) == 0:
        stop_layer_param_test.include.add()

    stop_layer_param_test.include[0].phase = caffe_pb2.TEST
    stop_layer_param_test.bottom[0] = lat_layer_name
    protobuf_insert_element_in_composite_container(stop_layer_train_index + 1, stop_layer_param_test, net_param.layer)

    # for i in range(stop_layer_train_index-2, stop_layer_train_index+3): # Debug
    #    print('At index', i, ':', net_param.layer[i])

    with open(result_net, 'w') as f:
        f.write(str(net_param))

    # print('Last:', net_param.layer[-1])
    # print(net_param)
    # dump(net_param.layer)
