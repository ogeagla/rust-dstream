extern crate nalgebra as na;

use dstream::*;
use na::*;

#[test]
fn initialize_clusters() {
    let result = initialize_clustering(vec!(Grid));
    assert_eq!(Ok(()), result);
}
