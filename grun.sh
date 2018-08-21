CUDA_VISIBLE_DEVICES=0 python3.6 CIFAR_MAIN.py --lr 0.1 --weight_thres 0.1 --loss_regu 0 --epoch 200 --print_freq 100 --log_file logs/log_quantized_ResNet20.txt --sigmoid_alpha 1 --quantize_layer1 2 --quantize_layer2 2 --quantize_layer3 2 --start_alpha 1 --end_alpha 5 --use_alpha_decay --use_quantize_weight

