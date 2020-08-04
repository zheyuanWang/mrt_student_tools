import tensorflow as tf

for event in tf.train.summary_iterator('/mrtstorage/users/zwang/eval/class19_tabal_initial4_eval/events.out.tfevents.1574094270.mrtknecht1'):
    for value in event.summary.value:
        print(value.tag)
        if value.HasField('simple_value'):
            print(value.simple_value)