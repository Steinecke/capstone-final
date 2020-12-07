from model import *

def main():
    
    ## train the model
#    data_dir = os.path.join(os.getcwd(),"data", "cs-train")
    model_train( data_dir = os.path.join(os.getcwd(), "data", "cs-train"), test=True)
    ## load the model
#    model = model_load()
    
    print("test run complete.")
if __name__ == "__main__":
    main()
