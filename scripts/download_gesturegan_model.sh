FILE=$1

echo "Note: available models are dayton_a2g_gesturegan_onecycle, dayton_g2a_gesturegan_onecycle, cvusa, dayton_a2g_64_gesturegan_onecycle, dayton_g2a_64_gesturegan_onecycle, ntu_gesturegan_twocycle and senz3d_gesturegan_twocycle"
echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/models/GestureGAN/${FILE}_pretrained.tar.gz
TAR_FILE=./checkpoints/${FILE}_pretrained.tar.gz
TARGET_DIR=./checkpoints/${FILE}_pretrained/

wget -N $URL -O $TAR_FILE

mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./checkpoints/
rm $TAR_FILE