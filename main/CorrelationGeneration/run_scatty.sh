"#!/bin/sh"

#for i in {1..5} 
#do 
#	echo "Run $i"
#	python MC_Model_Gen_weird_3D.py test $i >> test_log.txt
#done
#
echo "NAME h0l" >  scatty_config.txt 
echo "CENTRE 0 0 0" >> scatty_config.txt
echo "X_AXIS 8 0 0 128" >> scatty_config.txt
echo "Y_AXIS 0 0 8 128" >> scatty_config.txt
echo "Z_AXIS 0 0 0 1" >> scatty_config.txt
echo "WINDOW 3" >> scatty_config.txt
echo "SUM parallel" >> scatty_config.txt
echo "RADIATION X" >> scatty_config.txt
echo "REMOVE_BRAGG P" >> scatty_config.txt
./scatty test

echo "NAME hk0p5" >  scatty_config.txt 
echo "CENTRE 0 0 0.5" >> scatty_config.txt
echo "X_AXIS 8 0 0 128" >> scatty_config.txt
echo "Y_AXIS 0 8 0 128" >> scatty_config.txt
echo "Z_AXIS 0 0 0 1" >> scatty_config.txt
echo "WINDOW 3" >> scatty_config.txt
echo "SUM parallel" >> scatty_config.txt
echo "RADIATION X" >> scatty_config.txt
echo "REMOVE_BRAGG P" >> scatty_config.txt
./scatty test

echo "NAME 0kl" >  scatty_config.txt 
echo "CENTRE 0 0 0" >> scatty_config.txt
echo "X_AXIS 0 8 0 128" >> scatty_config.txt
echo "Y_AXIS 0 0 8 128" >> scatty_config.txt
echo "Z_AXIS 0 0 0 1" >> scatty_config.txt
echo "WINDOW 3" >> scatty_config.txt
echo "SUM parallel" >> scatty_config.txt
echo "RADIATION X" >> scatty_config.txt
echo "REMOVE_BRAGG P" >> scatty_config.txt
./scatty test

python plot.py test_h0l_sc.txt test_h0l.png 8 8
python plot.py test_0kl_sc.txt test_0kl.png 8 8
python plot.py test_hk0p5_sc.txt test_hk0p5.png 8 8


