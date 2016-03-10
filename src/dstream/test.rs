use dstream::{initialize_clustering};

#[test]
fn initialize_clusters() {
    let result = initialize_clustering();
    assert_eq!(Ok(()), result);
}
