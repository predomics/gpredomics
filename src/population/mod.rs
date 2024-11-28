use std::collections::HashMap;

mod data;

use data::Data;

pub struct Population {
    pub individuals: Vec<Individual>,
    pub data: &Data
}

impl Population {
    /// Provides a help message describing the `Population` struct and its fields.
    pub fn help() -> &'static str {
        "
        Population Struct:
        -----------------
        Represents a population consisting of multiple individuals, 
        with associated feature metadata.

        Fields:
        - individuals: Vec<Individual>
            A vector containing the individuals in the population. 
            Each individual represents an entity with a set of attributes or features.

        - feature_names: HashMap<u32, String>
            A map between feature indices (u32) and their corresponding names (String).
            This provides a human-readable label for each feature in the population.
        "
    }


    pub fn new(data: &Data) -> Population {
        Population {
            individuals: Vec::new(),
            data: data,
        }
    }


}