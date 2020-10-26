# Adapted from https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96
# Adapted from https://gist.github.com/fvisin/578089ae098424590d3f25567b6ee255
import sys
import tensorflow as tf
from datasets.Tool_read import cprint


V_HEAD_EX = "MobilenetV3/expanded_conv"
V_HEAD_CONV = "MobilenetV3/Conv"

V_REUSE_LAYER_ = "MobilenetV3/layer_"

V_PIPE2_EX = "MobilenetV3/pipe2/expanded_conv"
V_PIPE2_CONV = "MobilenetV3/pipe2/Conv"

V_UPPER_HEAD_EX = "MobilenetV3/upper_layers/expanded_conv"
V_UPPER_HEAD_CONV = "MobilenetV3/upper_layers/Conv"

V_PARALLEL_HEAD_EX = "MobilenetV3/parallel_layers/expanded_conv"
V_PARALLEL_HEAD_CONV = "MobilenetV3/parallel_layers/Conv"

def _report(var_name, new_name, style=0):
    print('{} ----->>'.format(var_name))
    cprint(new_name, 2+style)

def checkpoint_saver(new_var_pair_list, checkpoint,sess):
    # Save the variables
    cprint('saving to checkpoint file...', 7)
    saver = tf.compat.v1.train.Saver(var_list=new_var_pair_list)
    sess.run(tf.compat.v1.global_variables_initializer())
    saver.save(sess, checkpoint.model_checkpoint_path)


def find(checkpoint_dir, find_str):
    with tf.Session():
        negate = find_str.startswith('!')
        if negate:
            find_str = find_str[1:]
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            if negate and find_str not in var_name:
                print('%s missing from %s.' % (find_str, var_name))
            if not negate and find_str in var_name:
                print('Found %s in %s.' % (find_str, var_name))


def upper_layers_unit(var_name, n_to_substract):
    new_name = var_name
    # if need to rename, for expaned_conv_i
    for i in range(n_to_substract - 1):
        node_name = V_HEAD_EX + "_" + str(i + n_to_substract) + "/"
        if node_name in var_name:
            if i == 0:  # 1st ex-conv layer, without _i suffix
                to_node_name = V_UPPER_HEAD_EX + "/"
            else:
                to_node_name = V_UPPER_HEAD_EX + "_" + str(i) + "/"
            new_name = new_name.replace(node_name, to_node_name)
            _report(var_name, new_name)
    # if need to rename, for Conv_1 -> Conv
    node_name = V_HEAD_CONV + "_" + str(1) + "/"
    if node_name in var_name:
        to_node_name = V_UPPER_HEAD_CONV + "/"
        new_name = new_name.replace(node_name, to_node_name)
        _report(var_name, new_name)
    return new_name


def rename_upper_layers(checkpoint_dir, n_to_substract=8, dry_run=True):
    """ @wzy
    loop inside before saving, to aviod duplicatint variables when call the renmae function multiple times
    archived structure with upper & parallel scopes
    :param checkpoint_dir: root dir should contain file "checkpoint" which has path in it
    :param n_to_substract: set to 8: upper_layers namescope has now 8 top layers in it
        value !=8 not tested!
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir) # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])

        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name) # return A numpy ndarray with a copy of the value of this variable.

            # if need to rename
            new_name = upper_layers_unit(var_name, n_to_substract)

            # weather renamed or not, save to list
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)

        if not dry_run: # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)


def rename_parallel_layers(checkpoint_dir, number_parallel_layers, dry_run=True):
    """ @wzy
    archived structure with upper & parallel scopes
    :param checkpoint_dir: root dir should contain file "checkpoint" which has path in it
    :param n_to_substract: set to 8: upper_layers namescope has now 8 top layers in it
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    assert number_parallel_layers>0 & number_parallel_layers<9  # out of range not supported

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir)  # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])

        new_name_p = None
        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)  # return A numpy ndarray with a copy of the value of this variable.
            # Set the new name
            new_name = var_name

            # check the suffix _1 problem
            var_name_split = var_name.split('_', -1)[-1]
            if len(var_name_split) <4: cprint(" Warning, var_name_split[-1]={}".format(var_name_split), 0)

            # 1. if need to rename, for Conv -> Conv_1 & Conv
            node_name = V_HEAD_CONV + "/"  # exp_tensor_name:  MobilenetV3/Conv/BatchNorm/beta
            if node_name in var_name:
                to_node_name = V_PARALLEL_HEAD_CONV + "_" + str(1) + "/"  # Conv_1
                new_name = new_name.replace(node_name, to_node_name)

                new_name_p = var_name
                to_node_name_p =  V_PARALLEL_HEAD_CONV + "/"  #  Conv
                new_name_p = new_name_p.replace(node_name, to_node_name_p)

                _report(var_name, (new_name_p, new_name))

            # 2. if need to rename, for expaned_conv_i
            #for i in range(1,number_parallel_layers-1):  # 17-7-2=8, due to upper_layers=7, first 2 layers don't have suffix
            for i in range(8):  # 17-7-2=8, due to upper_layers=7, first 2 layers don't have suffix
                # exp. number_parallel_layers = 5, first 5 layers should be parallel, they are
                # con, ex, ex1, ex2, ex3, ex3
                if i == 0: # expaned_conv
                    node_name = V_HEAD_EX +"/"
                    if node_name in var_name:
                        # ex_1
                        to_node_name = V_PARALLEL_HEAD_EX + "_" + str(2 * i + 1) + "/"
                        new_name = new_name.replace(node_name, to_node_name)
                        # ex
                        new_name_p = var_name
                        to_node_name_p = V_PARALLEL_HEAD_EX + "/"
                        new_name_p = new_name_p.replace(node_name, to_node_name_p)

                        _report(var_name, (new_name_p, new_name))

                else:  # expaned_conv_i
                    node_name = V_HEAD_EX + "_" + str(i) + "/"
                    if i < number_parallel_layers-1: # for i in range(1,number_parallel_layers-1)
                        if node_name in var_name:
                            # ex_2n+1
                            to_node_name = V_PARALLEL_HEAD_EX + "_" + str(2 * i) + "/"
                            new_name = new_name.replace(node_name, to_node_name)
                            # ex_2n
                            new_name_p = var_name
                            to_node_name_p = V_PARALLEL_HEAD_EX + "_" + str(2 * i + 1) + "/"
                            new_name_p = new_name_p.replace(node_name, to_node_name_p)
                            _report(var_name, (new_name_p, new_name))
                    else:  # 3. if don't "rename", but only add prefix. for the rest of expaned_conv_i layers,
                                # such as layer ex7 when number_parallel_layers = 8
                        if node_name in var_name: #todo: the suffix Nr of blocks after combination
                            to_node_name = V_PARALLEL_HEAD_EX + "_" + str(number_parallel_layers -1 + i) + "/"
                            new_name = new_name.replace(node_name, to_node_name)
                            _report(var_name, new_name)


        # for each var_name in original CK, weather renamed or not, save to list
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)
            if new_name_p != None:
                new_var_p = tf.Variable(var, name=new_name_p)
                new_var_pair_list.append(new_var_p)
                new_name_p = None

        if not dry_run: # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)


def rename_both_up_and_para_layers(checkpoint_dir, number_parallel_layers, N_frames=2, n_to_substract=8, dry_run=True):
    """ @wzy
    archived structure with upper & parallel scopes
    :param checkpoint_dir: root dir should contain file "checkpoint" which has path in it
    :param n_to_substract: set to 8: upper_layers namescope has now 8 top layers in it
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    assert number_parallel_layers>0 & number_parallel_layers<9  # out of range not supported

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir)  # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])

        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)  # return A numpy ndarray with a copy of the value of this variable.

            #new_name = var_name  #debug @wzy: maybe cause problem by moving this init of new_name in unit function

            # 1. if need to rename  --- upper layers ---
            new_name = upper_layers_unit(var_name, n_to_substract, dry_run)  # return var_name if not renamed

            # 2. if need to rename, for Conv -> Conv & Conv_1
            node_name = V_HEAD_CONV + "/"  # exp_tensor_name:  MobilenetV3/Conv/BatchNorm/beta
            if node_name in var_name:
                to_node_name = V_PARALLEL_HEAD_CONV + "_" + str(1) + "/"  # Conv_1
                new_name = new_name.replace(node_name, to_node_name)

                new_name2 = var_name
                to_node_name2 =  V_PARALLEL_HEAD_CONV + "/"  #  Conv
                new_name2 = new_name2.replace(node_name, to_node_name2)
                new_var2 = tf.Variable(var, name=new_name2)
                new_var_pair_list.append(new_var2)
                if N_frames==2:
                    _report(var_name, (new_name2, new_name), style=1)
                elif N_frames==3:  # &
                    new_name3 = var_name
                    to_node_name3 = V_PARALLEL_HEAD_CONV + "_" + str(2) + "/" # Conv_2
                    new_name3 = new_name3.replace(node_name, to_node_name3)
                    new_var3 = tf.Variable(var, name=new_name3)
                    new_var_pair_list.append(new_var3)
                    _report(var_name, (new_name, new_name2, new_name3), style=1)

            # 3. if need to rename, for expaned_conv_i
            for i in range(8):  # 17-7-2=8, due to upper_layers=7, first 2 layers don't have suffix
                # exp. number_parallel_layers = 5, first 5 layers should be parallel, they are
                # con, ex, ex1, ex2, ex3, ex3
                if i == 0:  # expaned_conv (without suffix)
                    node_name = V_HEAD_EX +"/"
                    if node_name in var_name:
                        # ex (without suffix)
                        to_node_name = V_PARALLEL_HEAD_EX + "/"
                        new_name = new_name.replace(node_name, to_node_name)
                        # ex_1
                        new_name2 = var_name
                        to_node_name2 = V_PARALLEL_HEAD_EX + "_" + str(2*i+1) + "/"
                        new_name2 = new_name2.replace(node_name, to_node_name2)
                        new_var2 = tf.Variable(var, name=new_name2)
                        new_var_pair_list.append(new_var2)
                        if N_frames==2:
                            _report(var_name, (new_name2, new_name), style=2)
                        if N_frames==3:  # &
                            new_name3 = var_name
                            to_node_name3 = V_PARALLEL_HEAD_EX + "_" + str(3 * i + 2) + "/"  # ex_2
                            new_name3 = new_name3.replace(node_name, to_node_name3)
                            new_var3 = tf.Variable(var, name=new_name3)
                            new_var_pair_list.append(new_var3)
                            _report(var_name, (new_name, new_name2, new_name3), style=2)
                else:  # expaned_conv_i with suffix
                    node_name = V_HEAD_EX + "_" + str(i) + "/"
                    if i < number_parallel_layers-1: # for i in range(1,number_parallel_layers-1)
                        if node_name in var_name:
                            if N_frames == 2:
                                # ex_2n
                                to_node_name = V_PARALLEL_HEAD_EX + "_" + str(2 * i) + "/"
                                new_name = new_name.replace(node_name, to_node_name)
                                # ex_2n+1
                                new_name2 = var_name
                                to_node_name2 = V_PARALLEL_HEAD_EX + "_" + str(2 * i + 1) + "/"
                                new_name2 = new_name2.replace(node_name, to_node_name2)
                                new_var2 = tf.Variable(var, name=new_name2)
                                new_var_pair_list.append(new_var2)

                                _report(var_name, (new_name2, new_name),  style=-2)

                            elif N_frames==3:
                                # ex_3n
                                to_node_name = V_PARALLEL_HEAD_EX + "_" + str(3 * i) + "/"
                                new_name = new_name.replace(node_name, to_node_name)
                                # ex_3n+1
                                new_name2 = var_name
                                to_node_name2 = V_PARALLEL_HEAD_EX + "_" + str(3 * i + 1) + "/"
                                new_name2 = new_name2.replace(node_name, to_node_name2)
                                new_var2 = tf.Variable(var, name=new_name2)
                                new_var_pair_list.append(new_var2)
                                # ex_3n+2
                                new_name3 = var_name
                                to_node_name3 = V_PARALLEL_HEAD_EX + "_" + str(3 * i + 2) + "/"
                                new_name3 = new_name3.replace(node_name, to_node_name3)
                                new_var3 = tf.Variable(var, name=new_name3)
                                new_var_pair_list.append(new_var3)

                                _report(var_name, (new_name, new_name2, new_name3 ), style=-2)

                    else:  # 4. if don't expand to parallel pipes: for the rest of expaned_conv_i layers,
                                # such as layer ex7 when number_parallel_layers = 8
                        if node_name in var_name:
                            if N_frames == 2:
                                to_node_name = V_PARALLEL_HEAD_EX + "_" + str(number_parallel_layers -1 + i) + "/"
                                new_name = new_name.replace(node_name, to_node_name)
                                _report(var_name, new_name, style=-1)
                            elif N_frames == 3:
                                to_node_name = V_PARALLEL_HEAD_EX + "_" + str(2*number_parallel_layers - 2 + i) + "/"
                                new_name = new_name.replace(node_name, to_node_name)
                                _report(var_name, new_name, style=-1)

            # for each var_name in original CK, weather renamed or not, save to list
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)

        if not dry_run: # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)


def rename_pipe2(checkpoint_dir, number_parallel_layers, dry_run=True):
    """ @wzy
    copy and add the variable to a new variable scope with prefix "pipe"
    leave the orignal name scope unchanged
    :param checkpoint_dir: root dir should contain file "checkpoint" which has path in it
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir)  # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])  # list of all variables collected

        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir,
                                                     var_name)  # return A numpy ndarray with a copy of the value of this variable.
            # current name to check
            new_name = var_name
            new_name_p2 = None
            # 1. if need to copy to new parallel pipw
            node_name = V_HEAD_CONV + "/"  # exp_tensor_name:  MobilenetV3/Conv/BatchNorm/beta
            if node_name in var_name:
                to_node_name = V_PIPE2_CONV + "/"  # Conv
                new_name_p2 = new_name.replace(node_name, to_node_name)

                _report(var_name, new_name_p2, style=1)
            # 3. if need to rename, for expaned_conv_i
            for i in range(number_parallel_layers-1):
                if i == 0:  # expaned_conv (without suffix)
                    node_name = V_HEAD_EX + "/"
                    if node_name in var_name:
                        # ex (without suffix)
                        to_node_name = V_PIPE2_EX + "/"
                        new_name_p2 = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name_p2)

                else:  # expaned_conv_i with suffix
                    node_name = V_HEAD_EX + "_" + str(i) + "/"
                    if node_name in var_name:
                        # ex (with suffix)
                        to_node_name = V_PIPE2_EX + "_" + str(i) + "/"
                        new_name_p2 = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name_p2, style=2)
        # still in loop of the current variable name
            # save renamed to list
            if not new_name_p2 == None:
                new_var_p2 = tf.Variable(var, name=new_name_p2)
                new_var_pair_list.append(new_var_p2)
            # save un-renamed to list
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)

        if not dry_run:  # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)

def rename_for_reuse(checkpoint_dir, dry_run=True):
    """ @wzy
    all mobilenet blocks have prefix /layer{}/ in which {} is the total layer index,
    examples:
        MobilenetV3/Conv/             ->    MobilenetV3/layer_0/Conv/
        MobilenetV3/expanded_conv_14/ ->    MobilenetV3/layer_15/expanded_conv/
        MobilenetV3/Conv_1/           ->    MobilenetV3/layer_16/Conv/
    :param checkpoint_dir: root dir should contain file "checkpoint" which has path in it
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir)  # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])  # list of all variables collected

        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir,
                                                     var_name)  # return A numpy ndarray with a copy of the value of this variable.
            # current name to check
            new_name = var_name

            node_name = V_HEAD_CONV + "/"  # MobilenetV3/Conv/
            if node_name in var_name:
                to_node_name = V_REUSE_LAYER_ + str(0) + "/Conv/"
                new_name = new_name.replace(node_name, to_node_name)
                _report(var_name, new_name, style=1)
            node_name = V_HEAD_CONV + "_1/"  # MobilenetV3/Conv_1/
            if node_name in var_name:
                to_node_name = V_REUSE_LAYER_ + str(16) + "/Conv/"
                new_name = new_name.replace(node_name, to_node_name)
                _report(var_name, new_name, style=-2)

            for i in range(15):
                if i == 0:  # expaned_conv (without suffix)
                # MobilenetV3/expanded_conv/ ->    MobilenetV3/layer_1/expanded_conv/
                    node_name = V_HEAD_EX + "/"
                    if node_name in var_name:
                        to_node_name = V_REUSE_LAYER_ + str(i+1) + "/expanded_conv/"
                        new_name = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name)
                else:  # expaned_conv_i with suffix
                    node_name = V_HEAD_EX + "_" + str(i) + "/"
                    if node_name in var_name:
                        to_node_name = V_REUSE_LAYER_ + str(i+1) + "/expanded_conv/"
                        new_name = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name, style=-1)


        # still in loop of the current variable name
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)

        if not dry_run:  # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)


def rename_combine_ck_pipe2(checkpoint_dir, checkpoint_dir2, number_parallel_layers, dry_run=True):
    """ @wzy
    add the variable from CK2 to a new variable scope with prefix "pipe"
    leave the CK1's orignal name scope unchanged
    :param checkpoint_dir: root dir of baseline_range=1 (should contain file "checkpoint" which has path in it)
    :param checkpoint_dir2: root dir of baseline_range=10 (10 is just for example)
    :param dry_run: if Ture, wouldn't change the checkpoint
    """

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    checkpoint2 = tf.train.get_checkpoint_state(checkpoint_dir2)

    with tf.Session() as sess:
        new_var_pair_list = []
        var_name_list = []
        var_name_list_of_tuple = tf.contrib.framework.list_variables(checkpoint_dir)  # return list of tuple
        for t in var_name_list_of_tuple:
            var_name_list.append(t[0])  # list of all variables collected

        for var_name in var_name_list:
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)  # return A numpy ndarray with a copy of the value of this variable.
            # current name to check
            new_name = var_name
            new_name_p2 = None
            # 1. if need to copy to new parallel pipw
            node_name = V_HEAD_CONV + "/"  # exp_tensor_name:  MobilenetV3/Conv/BatchNorm/beta
            if node_name in var_name:
                to_node_name = V_PIPE2_CONV + "/"  # Conv
                new_name_p2 = new_name.replace(node_name, to_node_name)

                _report(var_name, new_name_p2, style=1)
            # 3. if need to rename, for expaned_conv_i
            for i in range(number_parallel_layers - 1):
                if i == 0:  # expaned_conv (without suffix)
                    node_name = V_HEAD_EX + "/"
                    if node_name in var_name:
                        # ex (without suffix)
                        to_node_name = V_PIPE2_EX + "/"
                        new_name_p2 = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name_p2)

                else:  # expaned_conv_i with suffix
                    node_name = V_HEAD_EX + "_" + str(i) + "/"
                    if node_name in var_name:
                        # ex (with suffix)
                        to_node_name = V_PIPE2_EX + "_" + str(i) + "/"
                        new_name_p2 = new_name.replace(node_name, to_node_name)
                        _report(var_name, new_name_p2, style=2)
            # still in loop of the current variable name
            # save renamed to list
            if not new_name_p2 == None:
                # use variables from CK2 to fill the pipe2
                var2 = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
                new_var_p2 = tf.Variable(var2, name=new_name_p2)
                new_var_pair_list.append(new_var_p2)
            # save un-renamed to list
            new_var = tf.Variable(var, name=new_name)
            new_var_pair_list.append(new_var)

        if not dry_run:  # Save the variables
            checkpoint_saver(new_var_pair_list, checkpoint, sess)


if __name__ == '__main__':
    V_PIPE2_EX = "MobilenetV3/pipe2/expanded_conv"
    v_PIPE2_CONV = "MobilenetV3/pipe2/Conv"
    ORIGIN_DIR = "/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_r1_origin_350k_test"
    CK_BASELINE_R12 = "/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/mobilenet_checkpoint/m3l_b4_range12_baseline1F/train"
    #rename_upper_layers(ORIGIN_DIR, dry_run=False)
    #rename_both_layers(ORIGIN_DIR, N_frames=2, number_parallel_layers=2, dry_run=True)
    #rename_pipe2(ORIGIN_DIR, number_parallel_layers=2, dry_run=False)
    #rename_for_reuse(ORIGIN_DIR,dry_run=False)
    rename_combine_ck_pipe2(ORIGIN_DIR,CK_BASELINE_R12,number_parallel_layers=11,dry_run=False)