import matplotlib.pyplot as plt
import pickle


def main():
    source1 = "300D6B_video_LSTM_"
    source2 = "300D6B_grocery_LSTM_"
    source3 = "300D6B_digital_music_LSTM_"
    source4 = "300D6B_clothing_LSTM_"
    source5 = "300D6B_electronics_LSTM_"


    with open("important_plot.pickle",
              'rb') as handle:
        diction = pickle.load(handle)
    for key, value in diction.items():
        print(key)
        #print(len(value[0]))
    ##TODO show the training accuracy
    plt.plot(diction['training'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('per batch')
    plt.legend(['training'], loc='upper left')
    #plt.legend(['50D', '100D', '200D', '300D'], loc='upper left')
    plt.show()
    
    plt.plot(diction['training_ep'])
    plt.title('training loss')
    plt.ylabel('loss')
    plt.xlabel('per epoch')
    plt.legend(['training'], loc='upper left')
    #plt.legend(['50D', '100D', '200D', '300D'], loc='upper left')
    plt.show()

    ##TODO show the testing accuracy
    plt.plot(diction['validate'])
    plt.plot(diction['testing'])
    plt.title('validate and testing loss')
    plt.ylabel('loss')
    plt.xlabel('per batch')
    plt.legend(['validate', 'testing'], loc='upper left')
    plt.show()

    plt.plot(diction['validate_ep'])
    plt.plot(diction['testing_ep'])
    plt.title('validate and testing loss')
    plt.ylabel('loss')
    plt.xlabel('per epoch')
    plt.legend(['validate', 'testing'], loc='upper left')
    plt.show()
    '''
    ##TODO show the training loss
    plt.plot(diction[source1 + 'loss'])
    plt.plot(diction[source2 + 'loss'])
    plt.plot(diction[source3 + 'loss'])
    plt.plot(diction[source4 + 'loss'])
    plt.plot(diction[source5 + 'loss'])
    plt.title('testing loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['50D', '100D', '200D', '300D'], loc='upper left')
    plt.legend(['video', 'grocery', 'digital_music', 'clothing','electronics'], loc='upper left')
    plt.show()
    ##TODO show the testing loss
    plt.plot(diction[source1 + 'val_loss'])
    plt.plot(diction[source2 + 'val_loss'])
    plt.plot(diction[source3 + 'val_loss'])
    plt.plot(diction[source4 + 'val_loss'])
    plt.plot(diction[source5 + 'val_loss'])
    plt.title('testing loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['50D', '100D', '200D', '300D'], loc='upper left')
    plt.legend(['video', 'grocery', 'digital_music', 'clothing','electronics'], loc='upper left')
    plt.show()
    # summarize history for loss
    '''



if __name__ == '__main__':
    main()
