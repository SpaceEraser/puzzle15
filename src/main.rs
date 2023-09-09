// #[cfg(not(target_env = "msvc"))]
// use tikv_jemallocator::Jemalloc;

// #[cfg(not(target_env = "msvc"))]
// #[global_allocator]
// static GLOBAL: Jemalloc = Jemalloc;

use puzzle15::*;
fn main() {
    tracing_subscriber::fmt::init();

    let start = std::time::Instant::now();

    // let board = Board::from_human_readable(
    //     &[
    //         &[15, 7, 12, 5, 14],
    //         &[2, 17, 22, 25, 24],
    //         &[21, 19, 8, 3, 9],
    //         &[4, 10, 1, 20, 23],
    //         &[11, 6, 13, 16, 18],
    //     ],
    // );

    let board = Board::from_human_readable(&[
        &[12, 13, 4, 7],
        &[10, 15, 6, 5],
        &[9, 3, 16, 14],
        &[1, 2, 8, 11],
    ]);

    // let board = Board::from_human_readable(&[&[4, 2, 7], &[6, 9, 5], &[8, 3, 1]]);

    println!("board:\n{board}");
    dbg!(solve_board(board));

    println!("elapsed = {:?}", start.elapsed());
}
