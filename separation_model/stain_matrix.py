import numpy as np

hemato_1 = (73, 119, 185.0)
eosin = (245, 143.0, 204)
safran = (248.0, 245, 152)
hemato_2 = (123, 153, 198)
Wgt = -np.log(np.array([hemato_1, eosin, safran, hemato_2]).T / 255)
Wgt2 = -np.log(np.array([hemato_1, eosin, safran]).T / 255)
# plt.imshow(np.exp(-Wgt.T.reshape((1,4,3))))
# plt.axis('off')
# plt.savefig('stain_matrix_Wgt.jpeg')
