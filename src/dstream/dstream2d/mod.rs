use na::*;
mod test;

pub struct Grid {
    i: usize,
    j: usize,
}

const C_M: f64     = 3.0;
const C_L: f64     = 0.8;
const LAMBDA: f64  = 0.998;
const BETA: f64    = 0.3;

pub fn initialize_clustering(grid_list: Mat2<Grid>) -> Result<(), String> {
    //update density of all grids in grid_list
    //assign each dense grid to a distinct cluster
    //label all other grids as NO_CLASS; bad grids!
    /*
    do until no change in cluster labels
          foreach cluster c
            foreach outside grid g of c
                foreach neighboring grid h of g
                    if h belongs to cluster c'
                        if |c| > |c'| label all grids in c' as c
                        else label all grids in c as c'
                    else if (h is translational) label h as in c
    end do
    */
    Ok(())
}

#[test]
fn initialize_clusters() {
    const loc_i: usize = 0 as usize;
    const loc_j: usize = 0 as usize;
    const G1: Grid = Grid {i: loc_i, j: loc_j};
    let result = initialize_clustering(Mat2::new(G1, G1, G1, G1));
    assert_eq!(Ok(()), result);
}


pub fn adjust_clustering(grid_list: Mat2<Grid>) -> Result<(), String> {
    //update the density of all grids in grid_list
    /*
    foreach grid g whose attribute (dense/sparse/transitional) is changed since last call to adjust_clustering()
        if g is a sparse grid
            delete g from its cluster c, label g as NO_CLASS
            if (c becomes unconnected) split c into two clusters
        else if g is a dense grid
            among all neighboring grids of g, find out the grid h whoe cluster c_h has the largest size
            if h is a dense grid
                if (g is labelled as NO_CLASS) label g as in c_h
                else if (g is in cluster c and |c| > |c_h|)
                    label all grids in c_h as in c
                else if (g is in cluster c and |c| <= |c_h|)
                    label all grids in c as in c_h
            else if h is a transitional grid
                if ((g is NO_CLASS) and (h is an outside grid if g is added to c_h)) label g as in c_h
                else if (g is in cluster c and |c| >= |c_h|)
                    move h from cluster c_h to c
        else if (g is transitional grid)
            among neighboring clusters of g, find the largest one c'
            satisfying that g is an outside grid if added to it and
            label g as in c'
    */

    Ok(())
}