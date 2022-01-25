# Download VG dataset

export MAIN_DIR=$PWD

mkdir data
cd data

wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tgz
tar -xzvf azcopy.tgz;
cp azcopy_linux_amd64_10.13.0/azcopy .
./azcopy copy 'https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/datasets/visualgenome/' $PWD --recursive
rm -rf ./azcopy azcopy.tgz azcopy_linux_amd64_10.13.0/
mv visualgenome/ VG/

cd $MAIN_DIR
