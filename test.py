from cryoio.mrc import writeMRC, readMRC


filename = 'Data/Beta/Particles/Falcon_2012_06_12-14_33_35_0.mrc'
filename = 'exp/Beta_sagd_noinit/model.mrc'
filename = 'Data/Beta/init.mrc'


m, hdr = readMRC(filename, inc_header=True)

print hdr