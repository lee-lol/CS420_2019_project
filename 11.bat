.\bin\convert_mnist_data.exe .\data\mnist\mnist_train_lmdb\train-images-idx3-ubyte .\data\mnist\mnist_train_lmdb\train-labels-idx1-ubyte .\examples\mnist\mnist_train_lmdb 

echo. 

.\bin\convert_mnist_data.exe .\data\mnist\mnist_test_lmdb\t10k-images-idx3-ubyte .\data\mnist\mnist_test_lmdb\t10k-labels-idx1-ubyte .\examples\mnist\mnist_test_lmdb

pause