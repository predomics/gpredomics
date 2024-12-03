use crate::data::Data;
use crate::population::Population;
use crate::param::Param;
use crate::individual::Individual;
use crate::utils;
use rand::{rngs::ThreadRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

/// This class implement Cross Validation dataset, e.g. split the Data in N folds and create N subset of Data each with its test subset.
pub struct CV {
    pub folds: Vec<Data>,
    pub datasets: Vec<Data>
}

impl CV {
    pub fn new(data: &Data, fold_number: usize, rng: &mut ChaCha8Rng) -> CV {

        let mut indices_class0:Vec<usize> = Vec::new();
        let mut indices_class1:Vec<usize> = Vec::new();
        for (i,f) in data.y.iter().enumerate() {
            if *f==0 { indices_class0.push(i) } else if *f==1 { indices_class1.push(i) }
        }

        let indices_class0_folds = utils::split_into_balanced_random_chunks(indices_class0, fold_number, rng);
        let indices_class1_folds = utils::split_into_balanced_random_chunks(indices_class1, fold_number, rng);

        let folds: Vec<Data>  = indices_class0_folds.into_iter().zip(indices_class1_folds.into_iter())
                    .map( |(i1,i2)| {i1.into_iter().chain(i2).collect::<Vec<usize>>()} )
                    .map( |i| { data.subset(i) } )
                    .collect();

        let mut datasets: Vec<Data> = Vec::new();
        for i in 0..fold_number {
            let mut dataset: Data = if i==0 { folds[1].clone() } else { folds[0].clone() };

            for j in 1..fold_number {
                if j==i { continue }
                else {
                    dataset.add(&folds[j]);
                }
            }

            datasets.push(dataset);
        }

        CV {
            folds: folds,
            datasets: datasets
        }
    }

    pub fn pass(&mut self, algo: fn(&mut Data, &Param) -> Vec<Population>, param: &Param) -> Vec<(Individual,f64,f64)> {

        let mut results_per_fold: Vec<(Individual,f64,f64)> = Vec::new();

        for (i,(train,test)) in self.datasets.iter_mut().zip(self.folds.iter_mut()).enumerate() {
            print!("Fold #{} : ",i+1);
            let mut best_model: Individual = algo(train, param).pop().unwrap().individuals.into_iter().take(1).next().unwrap();
            let train_auc = best_model.auc;
            let test_auc = best_model.compute_auc(test);

            println!("Train AUC: {:.3}  |  Test AUC: {:.3}", train_auc, test_auc);
            results_per_fold.push((best_model, train_auc, test_auc));
        }

        results_per_fold
    }
}

