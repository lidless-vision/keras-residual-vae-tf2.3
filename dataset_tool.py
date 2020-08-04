import tensorflow as tf
import pickle

"""
this is actually for the Tensorflow 2.3.0 version ??
"""

directory = '/mnt/md0/datasets/celeb-ms-cropped-aligned'
save_path = '/mnt/md0/datasets/ms-celeb-tf/'

build = True
use_compression = False

if build:
    print('building dataset from : ' + directory )

    my_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory, labels='inferred', label_mode='int', class_names=None,
        color_mode='rgb', batch_size=32, image_size=(64, 64), shuffle=False, seed=None,
        validation_split=None, subset=None, interpolation='bilinear', follow_links=False
    )

    print('saving tensorspec as pickle file.')
    print(my_dataset.element_spec)
    tensor_spec = my_dataset.element_spec
    pickle.dump(tensor_spec, open(save_path + "tensor_spec.pkl", "wb"))

    print('saving dataset to disk...')

    if use_compression:
        print('using gzip compression...')
        tf.data.experimental.save(dataset=my_dataset, path=save_path, compression='GZIP')
    else:
        print('saving dataset uncompressed')
        tf.data.experimental.save(dataset=my_dataset, path=save_path)

    print('dataset saved successfully?')

else:
    # load the dataset

    print('loading tensor_spec from pkl file')
    tensor_spec = pickle.load(open(save_path + "tensor_spec.pkl", "rb"))

    print('loading dataset from files.')
    loaded_dataset = tf.data.experimental.load(path=save_path, element_spec=tensor_spec, )
    #loaded_dataset = tf.data.experimental.load(path=save_path, compression='GZIP', element_spec=tensor_spec)

    print('caching dataset to file ')
    loaded_dataset = loaded_dataset.cache(save_path + 'cache_file.tf')

    print('loaded dataset')
    print(type(loaded_dataset))

    #print(loaded_dataset.batch(32, drop_remainder=True))



