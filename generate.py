import gpt_2_simple as gpt2

if __name__ == '__main__':
    gpt2.download_gpt2()

    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, 'forgotten_futures.txt', steps=1000)

    text = gpt2.generate(sess, return_as_list=True)[0]
    
    with open('output.txt', 'w') as f:
        f.write(text)
