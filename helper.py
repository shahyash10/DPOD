import pickle

# Pickle functions to save and load dictionaries
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# helper function to plot grpahs
def visualize(array):
    "Plot all images in the array of tensors in one row"
    for z in range(0,len(array)):
        temp = array[z]
        if temp.ndim > 3: # tensor output in the form NCHW
            temp = (torch.argmax(temp,dim=1).squeeze())
        if len(temp.shape) >= 3:
            plt.figure()
            plt.imshow(np.transpose(temp.detach().numpy().squeeze(),(1,2,0)))
            plt.show()
        else:
            plt.figure()
            plt.imshow(temp.detach().numpy(),cmap='gray')
            plt.show()