use rand_chacha::ChaCha8Rng;

use crate::cv::CV;
use crate::data::Data;
use crate::param::Param;

/* pub fn explore(data: &Data, algo: fn(&mut Data, &Param), param: &Param, hyperparameters: Vec<(String,Vec<T>)>, fold_number: usize, rng: &mut ChaCha8Rng) -> Vec<String,T> {
    let crossval = CV::new(data, fold_number, rng);
    for (hyperparam_name,hyperparam_values) in hyperparameters.iter() {
        let new_param = 
    } 

}
*/