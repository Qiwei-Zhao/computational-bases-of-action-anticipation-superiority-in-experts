# INPUT: Model Input Datas
# OUTPUT: Model Output

######################################### Parallel Processing ####################################################


import time
import readout_models
import encoding_models
import concurrent.futures as cf
import numpy as np

def run_readout_models(model_params):
    # Create thread pool with default number of processes
    with cf.ProcessPoolExecutor() as executor:
        # Start timer
        start_time = time.time()

        # Process parameter sets in parallel and store results in list
        results = list(executor.map(readout_models.run_readout, model_params))

        # End timer
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Save results to file
    #with open('results.pkl', 'wb') as f: 
    #    pickle.dump(results, f)

    return results


# model_params = (iC, iG, nsub, nresample, permtest, cv, verbose)
# 1. resampled_model x = (1, 1, 11, 200, False, False, 1)
# 2. permutation_model x = (1, 1, 11, 1, True, False, 1)
# 3. cross_validation_model x = (1, 1, 11, 1, False, True, 2)

# model_params = (iC, iG, nsub, nresample, permtest, cv, verbose)
# all models
model_params = [
    # resampled_model
    (1, 1, 18, 200, False, False, 1),
    (2, 1, 18, 200, False, False, 1),
    (1, 2, 18, 200, False, False, 1),
    (2, 2, 18, 200, False, False, 1),
    # permutation_model
    (1, 1, 18, 1, True, False, 1),
    (2, 1, 18, 1, True, False, 1),
    (1, 2, 18, 1, True, False, 1),
    (2, 2, 18, 1, True, False, 1),
    # cross_validation_model
    (1, 1, 18, 1, False, True, 2),
    (2, 1, 18, 1, False, True, 2),
    (1, 2, 18, 1, False, True, 2),
    (2, 2, 18, 1, False, True, 2)
]


# run the models
model_result = run_readout_models(model_params)
# save the results as .npy file
np.save('readout model results.npy', model_result)



## Encoding Models

def run_encoding_models(model_params):
    # Create thread pool with default number of processes
    with cf.ProcessPoolExecutor() as executor:
        # Start timer
        start_time = time.time()

        # Process parameter sets in parallel and store results in list
        results = list(executor.map(encoding_models.run_encoding, model_params))

        # End timer
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Save results to file
    #with open('results.pkl', 'wb') as f: 
    #    pickle.dump(results, f)

    return results


# model_params = (iC, nresample, permtest, cv, verbose, plots)
model_params_encoding = [
    (1, 1000, False, False, 1, True), #resampled_model
    (1, 1, False, True, 2, False), # cv model
    (1, 1, True, True, 1, False), # permutation model
]
# run the models
encoding_model_result = run_encoding_models(model_params_encoding)

# save the results as .npy file
np.save('encoding mdoel results.npy', encoding_model_result)