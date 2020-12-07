from model import *

def main():
    
    ## train the model
#    data_dir = os.path.join(os.getcwd(),"data", "cs-train")
    model_train( data_dir = os.path.join(os.getcwd(), "data", "cs-train"), test=False)
    ## load the model
#    model = model_load()
    
    print("model training complete.")

#    print(('http://127.0.0.1:5000/logs/train-test.log').text)

if __name__ == "__main__":
    main()
