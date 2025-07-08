use serde::{Deserialize, Serialize};
use crate::param::ImportanceAggregation;
use crate::data::Data;
use crate::{population, utils};
use crate::population::Population;
use crate::cv::CV;
use crate::param::Param;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ImportanceScope {
    // Importance can be computed on : an individual (oob), a population (FBM), a collection of population (between folds)
    Collection, // Collection of populations <=> Inter-folds FBM
    Population { id: usize }, // Intra-fold FBM (ID = Fold number)
    Individual { model_hash: u64 }, // Individual
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ImportanceType {
    OOB, 
    Coefficient, 
    PosteriorProbability 
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

#[derive(Serialize, Deserialize, Clone, Debug)]
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Experiment {
    pub id: String,
    pub timestamp: String,
    pub algorithm: String,
    pub parameters: Param,

    pub train_data: Data,
    pub test_data: Option<Data>,

    // Results 
    pub collection: Option<Vec<Population>>, // onky if keep_trace==true
    pub final_population: Option<Population>,
    pub importance_collection: Option<ImportanceCollection>,

    // Metadata
    pub execution_time: f64,
    pub cv_data: Option<CV>,
}

impl Experiment {
    pub fn save_auto<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
            let path = path.as_ref();
            let ext = path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();

            match ext.as_str() {
                "json" => self.save_json(path),
                "msgpack" | "mp" => self.save_messagepack(path),
                "bin" | "bincode" => self.save_bincode(path),
                _ => {
                    let json_path = path.with_extension("json");
                    self.save_json(json_path)
                }
            }
        }

        /// Save to JSON
        fn save_json<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let json = serde_json::to_string_pretty(self)?;
            std::fs::write(path, json)?;
            Ok(())
        }

        /// Save to messagepack (R and Rust compatible)
        fn save_messagepack<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let encoded = rmp_serde::to_vec(self)?;
            std::fs::write(path, encoded)?;
            Ok(())
        }

        /// Save as bincode (Rust compatible)
        fn save_bincode<P: AsRef<std::path::Path>>(
            &self,
            path: P,
        ) -> Result<(), Box<dyn std::error::Error>> {
            let encoded = bincode::serialize(self)?;
            std::fs::write(path, encoded)?;
            Ok(())
        }

    pub fn load_auto<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let ext = path.extension()
                     .and_then(|e| e.to_str())
                     .unwrap_or("")
                     .to_ascii_lowercase();

        match ext.as_str() {
            "json" => Self::load_json(path),
            "msgpack" | "mp" => Self::load_messagepack(path),
            "bin" | "bincode" => Self::load_bincode(path),
            _ => Self::load_with_fallback(path),
        }
    }

    fn load_json<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let experiment: Experiment = serde_json::from_str(&content)?;
        Ok(experiment)
    }

    fn load_messagepack<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = rmp_serde::from_slice(&bytes)?;
        Ok(experiment)
    }

    fn load_bincode<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let experiment: Experiment = bincode::deserialize(&bytes)?;
        Ok(experiment)
    }

    fn load_with_fallback<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        
        // Tentative JSON d'abord (format texte)
        if let Ok(experiment) = Self::load_json(path) {
            return Ok(experiment);
        }
        
        // Tentative MessagePack
        if let Ok(experiment) = Self::load_messagepack(path) {
            return Ok(experiment);
        }
        
        // Tentative Bincode
        if let Ok(experiment) = Self::load_bincode(path) {
            return Ok(experiment);
        }
        
        Err("Unable to load the experience".into())
    }

    pub fn display_results(&self) {
        println!("=== EXPERIMENT RESULTS ===");
        println!("ID: {}", self.id);
        println!("Algorithm: {}", self.algorithm);
        println!("Timestamp: {}", self.timestamp);
        println!("Execution time: {:.2}s", self.execution_time);

        if let Some(mut final_pop) = self.final_population.clone() {
            println!("{}", final_pop.display(&self.train_data, self.test_data.as_ref(), &self.parameters));
        } else {
            println!("No final population available");
        }
        
        if let Some(ref importance_collection) = self.importance_collection {
            if !importance_collection.importances.is_empty() {
                let top_features = importance_collection.get_top(0.1);
                println!("{}", top_features.display_feature_importance_terminal(&self.train_data, 10));
            }
        }
    }
}

// pub struct Court {
//     judges: Population,
//     penalties: Option<Vec<f64>>
    
// }

