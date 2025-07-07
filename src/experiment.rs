use serde::{Deserialize, Serialize};
use crate::param::ImportanceAggregation;
use crate::data::Data;
use crate::{population, utils};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ImportanceScope {
    // Importance can be computed on : an individual (oob), a population (FBM), a collection of population (between folds)
    Collection, // Collection of populations <=> Inter-folds FBM
    Population { id: usize }, // Intra-fold FBM (ID = Fold number)
    Individual { model_hash: u64 }, // Individual
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ImportanceType {
    OOB, // MDA
    Coefficient, //
    PosteriorProbability // MCMC 
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Importance {
    pub importance_type: ImportanceType,
    pub feature_idx: usize, 
    pub scope: ImportanceScope,
    pub aggreg_method: Option<ImportanceAggregation>,
    pub importance: f64, 
    pub is_scaled: bool,
    pub dispersion: f64,
    pub scope_pct: f64,
    pub direction: Option<usize>, // the associated class for MCMC & Coefficient
}

pub struct ImportanceCollection {
    pub importances: Vec<Importance>
}

impl ImportanceCollection {
    pub fn new() -> ImportanceCollection {
        ImportanceCollection {
            importances: Vec::new()
        }
    }

    // Return importances associated with a feature
    pub fn feature(&self, idx: usize) -> ImportanceCollection {
        let importances = self
            .importances
            .iter()
            .filter(|imp| imp.feature_idx == idx)
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    // Return importances associated with a scope and/or type
    pub fn filter(&self, scope: Option<ImportanceScope>, imp_type: Option<ImportanceType>,) -> ImportanceCollection {

        let importances = self.importances
            .iter()
            .filter(|imp| {

                let scope_ok = match &scope {
                    None => true,                                      
                    Some(ImportanceScope::Individual { .. })  =>
                        matches!(imp.scope, ImportanceScope::Individual { .. }),
                    Some(ImportanceScope::Population { .. })  =>
                        matches!(imp.scope, ImportanceScope::Population { .. }),
                    Some(ImportanceScope::Collection)        =>
                        matches!(imp.scope, ImportanceScope::Collection),
                };

                // ------ Filtre type ------
                let type_ok  = imp_type.as_ref().map_or(true, |t| imp.importance_type == *t);
                scope_ok && type_ok
            })
            .cloned()
            .collect();

        ImportanceCollection { importances }
    }

    pub fn get_top(&self, pct: f64) -> ImportanceCollection {
        assert!((0.0..=1.0).contains(&pct));
        let mut subset = self.importances.clone();
        subset.sort_unstable_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
        let keep = ((subset.len() as f64 * pct).ceil() as usize).max(1);
        subset.truncate(keep);
        ImportanceCollection { importances: subset }
    }
    pub fn display_feature_importance_terminal(&self, data: &Data, nb_features: usize) -> String {
            let mut map: std::collections::HashMap<usize, (f64, f64)> = std::collections::HashMap::new();
            let mut agg = ImportanceAggregation::Mean;

            for imp in &self.importances {
                if matches!(imp.scope, ImportanceScope::Collection) {
                    map.insert(imp.feature_idx, (imp.importance, imp.dispersion));
                    if let Some(a) = &imp.aggreg_method {
                        agg = a.clone();
                    }
                }
            }
            utils::display_feature_importance_terminal(data, &map, nb_features, &agg)
    }
}

// pub struct Court {
//     judges: Population,
//     penalties: Option<Vec<f64>>
    
// }

