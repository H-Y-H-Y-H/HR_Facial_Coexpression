import os
import shutil
import numpy as np

a = np.loadtxt('/Users/yuhang/Downloads/real_robodata/mouth1128_cmds.csv')

src_pth = '/Users/yuhang/Downloads/real_robodata/mouth1128/'

dst_pth = '/Users/yuhang/Downloads/real_robodata/mouth1129/'

# for i in range(1128):
#     src = src_pth+'%d.png'%i
#     dst = dst_pth+'%d.png'%(i+1)
#
#
#     shutil.copyfile(src, dst)

