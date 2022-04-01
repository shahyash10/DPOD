mkdir LineMOD_Dataset
cd LineMOD_Dataset

# http://campar.in.tum.de/Main/StefanHinterstoisser

wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/ape.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/benchviseblue.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/can.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cat.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/driller.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/duck.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/glue.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/holepuncher.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/iron.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/lamp.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/phone.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/cam.zip'
wget --no-check-certificate -c 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/eggbox.zip'

for file in `ls`; do unzip $file; done;
cd ..
wget --no-check-certificate - c 'http://images.cocodataset.org/zips/val2017.zip'
unzip val2017.zip
