# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import xlrd
import time
from six.moves import xrange
import numpy as np
import xlsxwriter
from metrics_all import Metrics
import argparse

batch_size = 16

text_size = 22
text_size2 = 21
image_size = 1365

text_layer1_size = 128  # 128
text_layer2_size = 512
image_layer1_size = 1024  # 1024 1024 1024
image_layer2_size = 1024
image_layer3_size = 1024
image_layer4_size = 512

feature_size = 128
attribute_layer1_size = 1024
attribute_layer2_size = 512

#model_path = './morning\\model10'  # model path

pairs_size = 2
zzh = 0.001
epsilon = 1e-3

margin1_score = 4.0
margin2_score = 2.0
margin3_score = 2.0


def load_trainset():
    print('loading...trainset')
    brand_text_train = np.load('./trainset/brand_text_train.npy')
    in_text_train = np.load('./trainset/in_text_train.npy')
    brand_image_train = np.load('./trainset/brand_image_train.npy')
    in_image_train = np.load('./trainset/in_image_train.npy')
    audience_brand_text_train = np.load('./audience_trainset/brand_text_train.npy')
    audience_in_text_train = np.load('./audience_trainset/in_text_train.npy')
    audience_brand_image_train = np.load('./audience_trainset/brand_image_train.npy')
    audience_in_image_train = np.load('./audience_trainset/in_image_train.npy')
    attribute_brand_train = np.load('./attribute_trainset_n2v/brand_attribute_train_n2v_new.npy')
    attribute_in_train = np.load('./attribute_trainset_n2v/in_attribute_train_n2v.npy')
    print('ok')
    return brand_text_train, in_text_train, brand_image_train, in_image_train, audience_brand_text_train, audience_in_text_train, \
           audience_brand_image_train, audience_in_image_train, attribute_brand_train, attribute_in_train


def load_testset():
    print('loading...testset')
    brand_text_test = np.load('./testset/brand_text_test.npy')
    in_text_test = np.load('./testset/in_text_test.npy')
    brand_image_test = np.load('./testset/brand_image_test.npy')
    in_image_test = np.load('./testset/in_image_test.npy')
    audience_brand_text_test = np.load('./audience_testset/brand_text_test.npy')
    audience_in_text_test = np.load('./audience_testset/in_text_test.npy')
    audience_brand_image_test = np.load('./audience_testset/brand_image_test.npy')
    audience_in_image_test = np.load('./audience_testset/in_image_test.npy')
    attribute_brand_test = np.load('./attribute_testset_n2v/brand_attribute_test_n2v_new.npy')
    attribute_in_test = np.load('./attribute_testset_n2v/in_attribute_test_n2v.npy')
    print('ok')
    return brand_text_test, in_text_test, brand_image_test, in_image_test, audience_brand_text_test, audience_in_text_test, \
           audience_brand_image_test, audience_in_image_test, attribute_brand_test, attribute_in_test


def load_testset2():
    print('loading...testset2')
    brand_text_test = np.load('./testset/brand_text_test_2.npy')
    in_text_test = np.load('./testset/in_text_test_2.npy')
    brand_image_test = np.load('/data/dataset_100_with_text_new/testset/brand_image_test_2.npy')
    in_image_test = np.load('./testset/in_image_test_2.npy')
    audience_brand_text_test = np.load('./audience_testset/brand_text_test_2.npy')
    audience_in_text_test = np.load('./audience_testset/in_text_test_2.npy')
    audience_brand_image_test = np.load('./audience_testset/brand_image_test_2.npy')
    audience_in_image_test = np.load('./audience_testset/in_image_test_2.npy')
    attribute_brand_test = np.load('./attribute_testset_n2v/brand_attribute_test_2_n2v_new.npy')
    attribute_in_test = np.load('./attribute_testset_n2v/in_attribute_test_2_n2v.npy')
    print('ok')
    return brand_text_test, in_text_test, brand_image_test, in_image_test, audience_brand_text_test, audience_in_text_test, \
           audience_brand_image_test, audience_in_image_test, attribute_brand_test, attribute_in_test


def save_recommendation_result(model_p, Ep, l_brand, l_in, l_ist, l_score, l_score2, l_score3, l_score4):
    file = './' + model_p + '/_' + str(Ep) + '.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet = workbook.add_worksheet(u'sheet1')

    index_n = 0
    for n in range(0, len(l_score)):
        worksheet.write(index_n, 0, l_brand[index_n])
        worksheet.write(index_n, 1, l_in[index_n])
        worksheet.write(index_n, 2, l_ist[index_n])
        worksheet.write(index_n, 3, l_score[index_n])
        worksheet.write(index_n, 4, l_score2[index_n])
        worksheet.write(index_n, 5, l_score3[index_n])
        worksheet.write(index_n, 6, l_score4[index_n])
        index_n += 1
    workbook.close()


def influencer_vectors_inputs():
    influencers_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                     text_size))
    influencers_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                      image_size))
    audience_influencers_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                              text_size2))
    audience_influencers_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                               image_size))
    attribute_influencers_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                                feature_size))
    return influencers_text_placeholder, influencers_image_placeholder, audience_influencers_text_placeholder, \
           audience_influencers_image_placeholder, attribute_influencers_image_placeholder


def brand_vector_inputs():
    brand_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                               text_size))
    brand_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                image_size))

    audience_brand_text_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                        text_size2))
    audience_brand_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                         image_size))
    attribute_brand_image_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                                          feature_size))
    return brand_text_placeholder, brand_image_placeholder, audience_brand_text_placeholder, \
           audience_brand_image_placeholder, attribute_brand_image_placeholder


def get_batch(brand_text_, brand_image_, in_text_, in_image_, audience_brand_text_, audience_brand_image_,
              audience_in_text_, \
              audience_in_image_, attribute_brand_, attribute_influencer_, step):
    if ((step + 1) * batch_size * pairs_size > len(brand_text_)):
        brand_text = brand_text_[step * batch_size * pairs_size:]
        brand_image = brand_image_[step * batch_size * pairs_size:]
        in_text = in_text_[step * batch_size * pairs_size:]
        in_image = in_image_[step * batch_size * pairs_size:]
        audience_brand_text = audience_brand_text_[step * batch_size * pairs_size:]
        audience_brand_image = audience_brand_image_[step * batch_size * pairs_size:]
        audience_in_text = audience_in_text_[step * batch_size * pairs_size:]
        audience_in_image = audience_in_image_[step * batch_size * pairs_size:]

        attribute_brand = attribute_brand_[step * batch_size * pairs_size:]
        attribute_influencer = attribute_influencer_[step * batch_size * pairs_size:]
    else:
        brand_text = brand_text_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        brand_image = brand_image_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        in_text = in_text_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        in_image = in_image_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]

        audience_brand_text = audience_brand_text_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        audience_brand_image = audience_brand_image_[
                               step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        audience_in_text = audience_in_text_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        audience_in_image = audience_in_image_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]

        attribute_brand = attribute_brand_[step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        attribute_influencer = attribute_influencer_[
                               step * batch_size * pairs_size:(step + 1) * batch_size * pairs_size]
        # print('label:',label_)
    return brand_text, brand_image, in_text, in_image, audience_brand_text, audience_brand_image, audience_in_text, audience_in_image, \
           attribute_brand, attribute_influencer


def fill_feed_dict_train(brand_text_train, brand_image_train, in_text_train, in_image_train
                         , brand_text_pl, brand_image_pl, in_text_pl, in_image_pl
                         , audience_brand_text_train, audience_brand_image_train, audience_in_text_train,
                         audience_in_image_train
                         , audience_brand_text_pl, audience_brand_image_pl, audience_in_text_pl, audience_in_image_pl
                         , attribute_brand_train, attribute_in_train, attribute_brand_train_pl, attribute_in_train_pl,
                         step, keep_prob, is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    brand_text_feed, brand_image_feed, in_text_feed, in_image_feed, audience_brand_text_feed, audience_brand_image_feed, audience_in_text_feed, \
    audience_in_image_feed, attribute_brand_feed, attribute_in_feed = get_batch(brand_text_train, brand_image_train,
                                                                                in_text_train, in_image_train \
                                                                                , audience_brand_text_train,
                                                                                audience_brand_image_train,
                                                                                audience_in_text_train, \
                                                                                audience_in_image_train,
                                                                                attribute_brand_train,
                                                                                attribute_in_train, step)
    feed_dict = {
        brands_text: brand_text_feed,
        brands_image: brand_image_feed,
        influencers_text: in_text_feed,
        influencers_image: in_image_feed,
        audience_brands_text: audience_brand_text_feed,
        audience_brands_image: audience_brand_image_feed,
        audience_influencers_text: audience_in_text_feed,
        audience_influencers_image: audience_in_image_feed,
        attribute_brands: attribute_brand_feed,
        attribute_influencers: attribute_in_feed,
        keep_prob: 0.5,
        is_training: True
    }
    return feed_dict


def fill_feed_dict_test(brand_text_train, brand_image_train, in_text_train, in_image_train
                        , brand_text_pl, brand_image_pl, in_text_pl, in_image_pl
                        , audience_brand_text_train, audience_brand_image_train, audience_in_text_train,
                        audience_in_image_train
                        , audience_brand_text_pl, audience_brand_image_pl, audience_in_text_pl, audience_in_image_pl
                        , attribute_brand_test, attribute_in_test, attribute_brand_test_pl, attribute_in_test_pl
                        , step, keep_prob, is_training):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    brand_text_feed, brand_image_feed, in_text_feed, in_image_feed, audience_brand_text_feed, audience_brand_image_feed, \
    audience_in_text_feed, audience_in_image_feed, attribute_brand_feed, attribute_in_feed = get_batch(brand_text_train,
                                                                                                       brand_image_train,
                                                                                                       in_text_train, \
                                                                                                       in_image_train,
                                                                                                       audience_brand_text_train,
                                                                                                       audience_brand_image_train, \
                                                                                                       audience_in_text_train,
                                                                                                       audience_in_image_train,
                                                                                                       attribute_brand_test,
                                                                                                       attribute_in_test,
                                                                                                       step)
    feed_dict = {brands_text: brand_text_feed,
                 brands_image: brand_image_feed,
                influencers_text: in_text_feed,
                influencers_image: in_image_feed,
                audience_brands_text: audience_brand_text_feed,
                audience_brands_image: audience_brand_image_feed,
                audience_influencers_text: audience_in_text_feed,
                audience_influencers_image: audience_in_image_feed,
                attribute_brands: attribute_brand_feed,
                attribute_influencers: attribute_in_feed,
                keep_prob: 1,
                is_training : False}
    return feed_dict


def get_weights(shape, lambd):
    var = tf.Variable(tf.random_normal(shape, stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(lambd)(var))
    return var


def get_attribute_representation(attribute_brands, attribute_influencers):
    w_11 = get_weights([feature_size, attribute_layer1_size], zzh)
    b_11 = tf.Variable(tf.random_normal([attribute_layer1_size], stddev=0.1))

    brand_attribute_representation_v1 = tf.nn.leaky_relu(tf.matmul(attribute_brands, w_11) + b_11, 0.01)
    in_attribute_representation_v1 = tf.nn.leaky_relu(tf.matmul(attribute_influencers, w_11) + b_11, 0.01)

    w_2 = get_weights([attribute_layer1_size, attribute_layer2_size], zzh)
    b_2 = tf.Variable(tf.random_normal([attribute_layer2_size], stddev=0.1))

    brand_attribute_representation_v2 = tf.matmul(brand_attribute_representation_v1, w_2) + b_2
    in_attribute_representation_v2 = tf.matmul(in_attribute_representation_v1, w_2) + b_2
    return brand_attribute_representation_v2, in_attribute_representation_v2


def get_content_representation(brands_text, brands_image, influencers_text, influencers_image,keep_prob):
    w_content_text1 = get_weights([text_size, text_layer1_size], zzh)
    dropout1 = tf.nn.dropout(w_content_text1, keep_prob)
    b_content_text1 = tf.Variable(tf.random_normal([text_layer1_size], stddev=0.1))

    brand_content_text_representation_v1 = tf.nn.leaky_relu(tf.matmul(brands_text, dropout1) + b_content_text1, 0.01)
    in_content_text_representation_v1 = tf.nn.leaky_relu(tf.matmul(influencers_text, dropout1) + b_content_text1, 0.01)

    w_content_text2 = get_weights([text_layer1_size, text_layer2_size], zzh)
    b_content_text2 = tf.Variable(tf.random_normal([text_layer2_size], stddev=0.1))

    brand_content_text_representation_v2 = tf.matmul(brand_content_text_representation_v1,
                                                     w_content_text2) + b_content_text2
    in_content_text_representation_v2 = tf.matmul(in_content_text_representation_v1, w_content_text2) + b_content_text2

    w_content_image1 = get_weights([image_size, image_layer1_size], zzh)
    dropout3 = tf.nn.dropout(w_content_image1, keep_prob)
    b_content_image1 = tf.Variable(tf.random_normal([image_layer1_size], stddev=0.1))

    brand_content_image_representation_v1 = tf.nn.leaky_relu(tf.matmul(brands_image, dropout3) + b_content_image1, 0.01)
    in_content_image_representation_v1 = tf.nn.leaky_relu(tf.matmul(influencers_image, dropout3) + b_content_image1,
                                                          0.01)

    w_content_image2 = get_weights([image_layer1_size, image_layer2_size], zzh)
    b_content_image2 = tf.Variable(tf.random_normal([image_layer2_size], stddev=0.1))

    brand_content_image_representation_v2 = tf.nn.leaky_relu(
        tf.matmul(brand_content_image_representation_v1, w_content_image2) + b_content_image2, 0.01)
    in_content_image_representation_v2 = tf.nn.leaky_relu(
        tf.matmul(in_content_image_representation_v1, w_content_image2) + b_content_image2, 0.01)

    w_content_image3 = get_weights([image_layer2_size, image_layer3_size], zzh)
    b_content_image3 = tf.Variable(tf.random_normal([image_layer3_size], stddev=0.1))

    brand_content_image_representation_v3 = tf.nn.leaky_relu(
        tf.matmul(brand_content_image_representation_v2, w_content_image3) + b_content_image3, 0.01)
    in_content_image_representation_v3 = tf.nn.leaky_relu(
        tf.matmul(in_content_image_representation_v2, w_content_image3) + b_content_image3, 0.01)

    w_content_image4 = get_weights([image_layer3_size, image_layer4_size], zzh)
    b_content_image4 = tf.Variable(tf.random_normal([image_layer4_size], stddev=0.1))

    brand_content_image_representation_v4 = tf.matmul(brand_content_image_representation_v3,
                                                      w_content_image4) + b_content_image4
    in_content_image_representation_v4 = tf.matmul(in_content_image_representation_v3,
                                                   w_content_image4) + b_content_image4

    brand_content_representation = tf.multiply(brand_content_text_representation_v2, brand_content_image_representation_v4)
    in_content_representation = tf.multiply(in_content_text_representation_v2, in_content_image_representation_v4)

    return brand_content_representation, in_content_representation


def get_audience_representation(audience_brands_text, audience_brands_image, audience_influencers_text,
                                audience_influencers_image,keep_prob):
    w_audience_text1 = get_weights([text_size2, text_layer1_size], zzh)
    dropout2 = tf.nn.dropout(w_audience_text1, keep_prob)
    b_audience_text1 = tf.Variable(tf.random_normal([text_layer1_size], stddev=0.1))

    brand_au_text_representation_v1 = tf.nn.leaky_relu(tf.matmul(audience_brands_text, dropout2) + b_audience_text1,
                                                       0.01)
    in_au_text_representation_v1 = tf.nn.leaky_relu(tf.matmul(audience_influencers_text, dropout2) + b_audience_text1,
                                                    0.01)

    w_audience_text2 = get_weights([text_layer1_size, text_layer2_size], zzh)
    b_audience_text2 = tf.Variable(tf.random_normal([text_layer2_size], stddev=0.1))

    brand_au_text_representation_v2 = tf.matmul(brand_au_text_representation_v1, w_audience_text2) + b_audience_text2
    in_au_text_representation_v2 = tf.matmul(in_au_text_representation_v1, w_audience_text2) + b_audience_text2

    w_audience_image1 = get_weights([image_size, image_layer1_size], zzh)
    dropout4 = tf.nn.dropout(w_audience_image1, keep_prob)
    b_audience_image1 = tf.Variable(tf.random_normal([image_layer1_size], stddev=0.1))

    brand_au_image_representation_v1 = tf.nn.leaky_relu(tf.matmul(audience_brands_image, dropout4) + b_audience_image1,
                                                        0.01)
    in_au_image_representation_v1 = tf.nn.leaky_relu(
        tf.matmul(audience_influencers_image, dropout4) + b_audience_image1, 0.01)

    w_audience_image2 = get_weights([image_layer1_size, image_layer2_size], zzh)
    b_audience_image2 = tf.Variable(tf.random_normal([image_layer2_size], stddev=0.1))

    brand_au_image_representation_v2 = tf.nn.leaky_relu(
        tf.matmul(brand_au_image_representation_v1, w_audience_image2) + b_audience_image2, 0.01)
    in_au_image_representation_v2 = tf.nn.leaky_relu(
        tf.matmul(in_au_image_representation_v1, w_audience_image2) + b_audience_image2, 0.01)

    w_audience_image3 = get_weights([image_layer2_size, image_layer3_size], zzh)
    b_audience_image3 = tf.Variable(tf.random_normal([image_layer3_size], stddev=0.1))

    brand_au_image_representation_v3 = tf.nn.leaky_relu(
        tf.matmul(brand_au_image_representation_v2, w_audience_image3) + b_audience_image3, 0.01)
    in_au_image_representation_v3 = tf.nn.leaky_relu(
        tf.matmul(in_au_image_representation_v2, w_audience_image3) + b_audience_image3, 0.01)

    w_audience_image4 = get_weights([image_layer3_size, image_layer4_size], zzh)
    b_audience_image4 = tf.Variable(tf.random_normal([image_layer4_size], stddev=0.1))

    brand_au_image_representation_v4 = tf.matmul(brand_au_image_representation_v3, w_audience_image4) + b_audience_image4
    in_au_image_representation_v4 = tf.matmul(in_au_image_representation_v3, w_audience_image4) + b_audience_image4

    brand_au_representation = tf.multiply(brand_au_text_representation_v2, brand_au_image_representation_v4)
    in_au_representation = tf.multiply(in_au_text_representation_v2, in_au_image_representation_v4)

    return brand_au_representation, in_au_representation


graph1 = tf.Graph()
with graph1.as_default():
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, name='training')
    influencers_text, influencers_image, audience_influencers_text, audience_influencers_image, \
    attribute_influencers = influencer_vectors_inputs()
    brands_text, brands_image, audience_brands_text, audience_brands_image, attribute_brands = brand_vector_inputs()
    ############################################################
    brand_attribute_representation_v2, in_attribute_representation_v2 = get_attribute_representation(attribute_brands, attribute_influencers)
    ############################################################
    brand_content_representation, in_content_representation = get_content_representation(brands_text, brands_image, influencers_text, influencers_image,keep_prob)
    brand_au_representation, in_au_representation = get_audience_representation(audience_brands_text, audience_brands_image, audience_influencers_text, audience_influencers_image,keep_prob)
    ############################################################
    # co-attention mechanism   512 is dimension length
    w_brand_fusion1 = get_weights([image_layer4_size, 512], zzh)
    w_brand_fusion2 = get_weights([image_layer4_size, 512], zzh)
    w_in_fusion1 = get_weights([image_layer4_size, 512], zzh)
    w_in_fusion2 = get_weights([image_layer4_size, 512], zzh)
    b_1 = tf.Variable(tf.random_normal([512], stddev=1))
    b_2 = tf.Variable(tf.random_normal([512], stddev=1))
    b_3 = tf.Variable(tf.random_normal([512], stddev=1))
    b_4 = tf.Variable(tf.random_normal([512], stddev=1))

    brand_content_representation_v2 = tf.nn.sigmoid(tf.matmul(brand_content_representation, w_brand_fusion1) + b_1)
    brand_au_representation_v2 = tf.nn.sigmoid(tf.matmul(brand_au_representation, w_brand_fusion2) + b_2)

    in_content_representation_v2 = tf.nn.sigmoid(tf.matmul(in_content_representation, w_in_fusion1) + b_3)
    in_au_representation_v2 = tf.nn.sigmoid(tf.matmul(in_au_representation, w_in_fusion2) + b_4)
    # affinity representation
    brand_affinity_representation = brand_content_representation_v2 * brand_au_representation_v2
    in_affinity_representation = in_content_representation_v2 * in_au_representation_v2

    weighted1 = tf.nn.softmax(brand_affinity_representation * tf.nn.sigmoid(brand_content_representation))
    weighted2 = tf.nn.softmax(in_affinity_representation * tf.nn.sigmoid(in_content_representation))
    weighted3 = tf.nn.softmax(brand_affinity_representation * tf.nn.sigmoid(brand_au_representation))
    weighted4 = tf.nn.softmax(in_affinity_representation * tf.nn.sigmoid(in_au_representation))

    brand_content_representation = weighted1 * brand_content_representation * 512
    in_content_representation = weighted2 * in_content_representation * 512
    brand_au_representation = weighted3 * brand_au_representation * 512
    in_au_representation = weighted4 * in_au_representation * 512
    ###########################################################################
    # ranking sub-function f
    sim = tf.multiply(brand_content_representation, in_content_representation)
    sim = tf.reduce_sum(sim, axis=1)
    # ranking sub-function g
    audience_sim = tf.multiply(brand_au_representation, in_au_representation)
    audience_sim = tf.reduce_sum(audience_sim, axis=1)
    # ranking sub-function h
    attribute_sim = tf.multiply(brand_attribute_representation_v2, in_attribute_representation_v2)
    attribute_sim = tf.reduce_sum(attribute_sim, axis=1)

    # ------------------
    sim = tf.reshape(sim, [batch_size, pairs_size])
    score = tf.transpose(sim)
    anchor_positive_score, anchor_negative_score = tf.split(score, [1, 1], 0)

    audience_sim = tf.reshape(audience_sim, [batch_size, pairs_size])
    audience_score = tf.transpose(audience_sim)
    audience_anchor_positive_score, audience_anchor_negative_score = tf.split(audience_score, [1, 1], 0)

    attribute_sim = tf.reshape(attribute_sim, [batch_size, pairs_size])
    attribute_score = tf.transpose(attribute_sim)
    attribute_anchor_positive_score, attribute_anchor_negative_score = tf.split(attribute_score, [1, 1], 0)

    anchor_positive = tf.reshape(anchor_positive_score, [batch_size])
    anchor_negative = tf.reshape(anchor_negative_score, [batch_size])

    audience_anchor_positive = tf.reshape(audience_anchor_positive_score, [batch_size])
    audience_anchor_negative = tf.reshape(audience_anchor_negative_score, [batch_size])

    attribute_anchor_positive = tf.reshape(attribute_anchor_positive_score, [batch_size])
    attribute_anchor_negative = tf.reshape(attribute_anchor_negative_score, [batch_size])

    all_anchor_positive = anchor_positive + audience_anchor_positive + attribute_anchor_positive
    all_anchor_negative = anchor_negative + audience_anchor_negative + attribute_anchor_negative
    # ------------------------
    x1 = tf.ones([1, batch_size])
    x2 = tf.zeros([1, batch_size])
    label = tf.concat([x1, x2], 0)
    label = tf.transpose(label)

    # Global ranking function F
    all_sim = sim + audience_sim + attribute_sim
    all_sim = tf.nn.softmax(all_sim)
    all_sim = all_sim + 1e-16

    # global loss
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(label * tf.log(all_sim), axis=1))

    all_anchor_positive_score = anchor_positive_score + audience_anchor_positive_score + attribute_anchor_positive_score
    all_anchor_negative_score = anchor_negative_score + audience_anchor_negative_score + attribute_anchor_negative_score
    # triplet_loss1
    margin1 = tf.ones([1, batch_size]) * margin1_score
    triplet_loss1 = anchor_negative_score - anchor_positive_score + margin1
    triplet_loss1 = tf.maximum(triplet_loss1, 0.0)
    valid_triplets1 = tf.to_float(tf.greater(triplet_loss1, 1e-16))
    num_positive_triplets1 = tf.reduce_sum(valid_triplets1)
    triplet_loss1 = tf.reduce_sum(triplet_loss1) / (num_positive_triplets1 + 1e-16)
    # triplet_loss2
    margin2 = tf.ones([1, batch_size]) * margin2_score
    triplet_loss2 = audience_anchor_negative_score - audience_anchor_positive_score + margin2
    triplet_loss2 = tf.maximum(triplet_loss2, 0.0)
    valid_triplets2 = tf.to_float(tf.greater(triplet_loss2, 1e-16))
    num_positive_triplets2 = tf.reduce_sum(valid_triplets2)
    triplet_loss2 = tf.reduce_sum(triplet_loss2) / (num_positive_triplets2 + 1e-16)
    # triplet_loss3
    margin3 = tf.ones([1, batch_size]) * margin3_score
    triplet_loss3 = attribute_anchor_negative_score - attribute_anchor_positive_score + margin3
    triplet_loss3 = tf.maximum(triplet_loss3, 0.0)
    valid_triplets3 = tf.to_float(tf.greater(triplet_loss3, 1e-16))
    num_positive_triplets3 = tf.reduce_sum(valid_triplets3)
    triplet_loss3 = tf.reduce_sum(triplet_loss3) / (num_positive_triplets3 + 1e-16)

    regularization = tf.reduce_mean(anchor_positive) * 0.01 + tf.reduce_mean(anchor_negative) * 0.01 + tf.reduce_mean(
        audience_anchor_positive) * 0.02 + \
            tf.reduce_mean(audience_anchor_negative) * 0.02 + tf.reduce_mean(
        attribute_anchor_positive) * 0.01 + tf.reduce_mean(attribute_anchor_negative) * 0.01

    LEARNING_RATE_BASE = 0.002
    LEARNING_RATE_DECAY = 0.99
    LEARNING_RATE_STEP = 12000
    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_steps, LEARNING_RATE_STEP,
                                               LEARNING_RATE_DECAY, staircase=True)
    tf.add_to_collection('losses', cross_entropy)
    tf.add_to_collection('losses', triplet_loss1)
    tf.add_to_collection('losses', triplet_loss2)
    tf.add_to_collection('losses', triplet_loss3)
    tf.add_to_collection('losses', regularization)
    loss = tf.add_n(tf.get_collection('losses'))
    #train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_steps)

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-d", "--dataset", required=True, help="path to input dataset (i.e., directory of images)")
    ap.add_argument("-m", "--model", required=True, help="path to output model")
    args = vars(ap.parse_args())

    Epoch_ = 100
    Step_ = 0
    Epoch = 0
    Part = 1  # the number of training set parts
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.visible_device_list = "0"

    # metrics
    l_r_10 = []
    l_r_50 = []
    l_medr = []
    l_auc = []
    l_cauc = []
    l_mrr = []
    l_map = []
    with tf.Session(graph=graph1, config=config) as sess:

        saver = tf.train.Saver(max_to_keep=15)

        init = tf.global_variables_initializer()
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(args["model"])
        if ckpt and ckpt.all_model_checkpoint_paths:
            path_ = ''
            for path in ckpt.all_model_checkpoint_paths:
                path_ = path
            print(path_)
            saver.restore(sess, path_)

        # ------------------------------loading data------------------------------------------------
        brand_text_train, in_text_train, brand_image_train, in_image_train, audience_brand_text_train, audience_in_text_train, \
        audience_brand_image_train, audience_in_image_train, attribute_brand_train, attribute_in_train = load_trainset()

        brand_text_test, in_text_test, brand_image_test, in_image_test, audience_brand_text_test, audience_in_text_test, \
        audience_brand_image_test, audience_in_image_test, attribute_brand_test, attribute_in_test = load_testset()

        brand_text_test2, in_text_test2, brand_image_test2, in_image_test2, audience_brand_text_test2, audience_in_text_test2, \
        audience_brand_image_test2, audience_in_image_test2, attribute_brand_test2, attribute_in_test2 = load_testset2()

        for j in range(0, Epoch_):
            print('-----train-----')
            print('Epoch %d' % (Epoch))
            for part in range(0, Part):
                if (len(brand_text_train) % (pairs_size * batch_size) == 0):
                    Step_ = len(brand_text_train) / (pairs_size * batch_size)
                else:
                    Step_ = int(len(brand_text_train) / (pairs_size * batch_size))
                Step_ = int(Step_)
                mean_loss = 0
                for step in xrange(Step_):
                    start_time = time.time()
                    feed_dict = fill_feed_dict_train(brand_text_train, brand_image_train, in_text_train, in_image_train
                                                     , brands_text, brands_image, influencers_text, influencers_image
                                                     , audience_brand_text_train, audience_brand_image_train,
                                                     audience_in_text_train, audience_in_image_train
                                                     , audience_brands_text, audience_brands_image,
                                                     audience_influencers_text, audience_influencers_image
                                                     , attribute_brand_train, attribute_in_train, attribute_brands,
                                                     attribute_influencers
                                                     , step, keep_prob, is_training)
                    _, _, ce_loss, t_loss1, t_loss2, t_loss3, l3, is_train = sess.run(
                        [brand_attribute_representation_v2, audience_sim, cross_entropy, triplet_loss1, triplet_loss2,
                         triplet_loss3, regularization, is_training], feed_dict=feed_dict)

                    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                    mean_loss += loss_value
                    duration = time.time() - start_time
                    if (step % 5000 == 0 and step != 0):
                        #lr, gs = sess.run([learning_rate, global_steps], feed_dict=feed_dict)
                        #print(lr, gs)
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, mean_loss / step, duration))
                        print('cross_entropy', ce_loss)
                        print('triplet_loss1', t_loss1)
                        print('triplet_loss2', t_loss2)
                        print('triplet_loss3', t_loss3)
                        print('regularization', l3)
            print('is_training:', is_train)
            globalstep = Epoch
            checkpoint_file = os.path.join(args["model"], 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=globalstep)

            print('-----test-----')
            if (len(brand_text_test) % (pairs_size * batch_size) == 0):
                tStep_ = len(brand_text_test) / (pairs_size * batch_size)
            else:
                tStep_ = int(len(brand_text_test) / (pairs_size * batch_size))
            tStep_ = int(tStep_)
            test_mean_loss = 0.0
            for t in xrange(tStep_):
                feed_dict = fill_feed_dict_test(brand_text_test, brand_image_test, in_text_test, in_image_test
                                                , brands_text, brands_image, influencers_text, influencers_image
                                                , audience_brand_text_test, audience_brand_image_test,
                                                audience_in_text_test, audience_in_image_test
                                                , audience_brands_text, audience_brands_image,
                                                audience_influencers_text, audience_influencers_image
                                                , attribute_brand_test, attribute_in_test, attribute_brands,
                                                attribute_influencers
                                                , t, keep_prob, is_training)
                test_loss, is_train = sess.run([loss, is_training], feed_dict=feed_dict)
                test_mean_loss += test_loss
                if (t % 1500 == 0 and t != 0):
                    print('Step %d: loss = %.2f ' % (t, test_mean_loss / t))
            print('is_training:', is_train)
            print('-----test2-----')
            ExcelFile1 = xlrd.open_workbook('./testset_r.xlsx')
            sheet1 = ExcelFile1.sheet_by_index(0)
            l_brand = []
            l_in = []
            l_ist = []
            l_score = []
            l_score2 = []
            l_score3 = []
            l_score4 = []
            index = 0
            if (len(brand_text_test2) % (pairs_size * batch_size) == 0):
                ttStep_ = len(brand_text_test2) / (pairs_size * batch_size)
            else:
                ttStep_ = int(len(brand_text_test2) / (pairs_size * batch_size))
            ttStep_ = int(ttStep_)
            test_mean_loss = 0.0
            for t in xrange(ttStep_):
                feed_dict = fill_feed_dict_test(brand_text_test2, brand_image_test2, in_text_test2, in_image_test2
                                                , brands_text, brands_image, influencers_text, influencers_image
                                                , audience_brand_text_test2, audience_brand_image_test2,
                                                audience_in_text_test2, audience_in_image_test2
                                                , audience_brands_text, audience_brands_image,
                                                audience_influencers_text, audience_influencers_image
                                                , attribute_brand_test2, attribute_in_test2, attribute_brands,
                                                attribute_influencers
                                                , t, keep_prob, is_training)
                score1, score2, score3, score4, test_loss, is_train = sess.run(
                    [all_anchor_positive, anchor_positive, audience_anchor_positive, attribute_anchor_positive, loss,
                     is_training],
                    feed_dict=feed_dict)
                test_mean_loss += test_loss
                for xyz in range(len(score1)):
                    xx = score1[xyz]
                    xy = score2[xyz]
                    xz = score3[xyz]
                    xxx = score4[xyz]
                    brand = sheet1.cell(index, 0).value.encode('utf-8').decode('utf-8-sig')
                    influencer = sheet1.cell(index, 1).value.encode('utf-8').decode('utf-8-sig')
                    ist = sheet1.cell(index, 2).value
                    l_brand.append(brand)
                    l_in.append(influencer)
                    l_ist.append(ist)
                    l_score.append(xx)
                    l_score2.append(xy)
                    l_score3.append(xz)
                    l_score4.append(xxx)
                    index += 1
            print('is_training:', is_train)
            # --------------Measurement-----------------------------------
            medr, r10, r50 = Metrics.metrics(l_brand, l_in, l_ist, l_score)
            auc, cauc = Metrics.auc(l_brand, l_in, l_ist, l_score)
            mrr = Metrics.mrr(l_brand, l_in, l_ist, l_score)
            map = Metrics.map(l_brand, l_in, l_ist, l_score)

            l_medr.append(medr)
            l_r_10.append(r10)
            l_r_50.append(r50)
            l_auc.append(auc)
            l_cauc.append(cauc)
            l_mrr.append(mrr)
            l_map.append(map)

            save_recommendation_result(args["model"], Epoch, l_brand, l_in, l_ist, l_score, l_score2, l_score3,l_score4)
            print(l_r_10[Epoch], l_r_50[Epoch], l_medr[Epoch], l_auc[Epoch], l_cauc[Epoch], l_mrr[Epoch], l_map[Epoch])
            Epoch += 1
    for ii in range(0, Epoch_):
        print(ii)
        print(l_r_10[ii], l_r_50[ii], l_medr[ii], l_auc[ii], l_cauc[ii], l_mrr[ii], l_map[ii])

if __name__ == '__main__':
    main()