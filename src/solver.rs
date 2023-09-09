use crate::PackedArray2D;
use fnv::FnvHashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use std::{
    collections::hash_map::{DefaultHasher, Entry},
    collections::BinaryHeap,
    hash::{Hash, Hasher},
};

#[tracing::instrument]
pub fn solve_board(initial_board: Board) -> Option<Vec<Direction>> {
    if !initial_board.has_solution() {
        return None;
    }

    let solution_board = Board::new(initial_board.width(), initial_board.height());
    let mut graph = DiGraph::<BoardNode, Direction>::default();
    let mut board_hash_to_node = FnvHashMap::default();
    let mut open_list = BinaryHeap::default();

    let initial_board_hash = hash(&initial_board);
    let initial_node_index = graph.add_node(BoardNode::new(initial_board, &solution_board));
    board_hash_to_node.insert(initial_board_hash, initial_node_index);

    graph[initial_node_index].is_in_open_list = true;
    graph[initial_node_index].dist_from_initial = Some(0);
    open_list.push(OpenListEntry {
        node_index: initial_node_index,
        score: graph[initial_node_index].f_score(),
    });

    // println!("Initial graph:\n{:?}", petgraph::dot::Dot::new(&graph));

    // A*
    while let Some(OpenListEntry {
        node_index: current_node_index,
        ..
    }) = open_list.pop()
    {
        graph[current_node_index].is_in_open_list = false;

        tracing::trace!("graph_node_count: {}", graph.node_count());

        // println!("Iterating:\n{:?}", petgraph::dot::Dot::new(&graph));

        let current_board = graph[current_node_index].get_board(&graph, current_node_index);
        // let current_board = graph[current_node_index].board.as_ref().cloned().unwrap();
        // graph[current_node_index].clear_board();

        // println!("Current board:\n{:?}\n", graph[current_node_index]);

        if *current_board == solution_board {
            break;
        }

        let neighbors = current_board
            .possible_moves()
            .into_iter()
            .map(|d| {
                let mut new_board = (*current_board).clone();
                new_board.swap_empty_tile(d);
                (d, new_board)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|(d, n)| {
                let n_hash = hash(&n);
                let n_index = match board_hash_to_node.entry(n_hash) {
                    Entry::Vacant(entry) => {
                        let n_node = BoardNode::new(n, &solution_board);
                        let n_index = graph.add_node(n_node);
                        entry.insert(n_index);
                        n_index
                    }
                    Entry::Occupied(entry) => *entry.get(),
                };

                graph.update_edge(current_node_index, n_index, d);

                n_index
            })
            .collect::<Vec<_>>();

        for n in neighbors {
            let tentative_g_score = graph[current_node_index]
                .dist_from_initial
                .map_or(u16::MAX, |dist_from_initial| dist_from_initial + 1);
            let current_g_score = graph[n].dist_from_initial.unwrap_or(u16::MAX);

            if tentative_g_score < current_g_score {
                graph[n].parent = Some(current_node_index);
                graph[n].dist_from_initial = Some(tentative_g_score);

                if !graph[n].is_in_open_list {
                    graph[n].is_in_open_list = true;
                    open_list.push(OpenListEntry {
                        node_index: n,
                        score: graph[n].f_score(),
                    });
                }
            }
        }
    }

    // println!("Backtracking...");

    // Backtrack
    let directions = {
        let mut reverse_directions = Vec::new();
        let mut current_node_index = *board_hash_to_node
            .get(&hash(&solution_board))
            .expect("Search finished but solution not found");

        while let Some(parent_index) = graph[current_node_index].parent {
            let edge_index = graph
                .find_edge(parent_index, current_node_index)
                .expect("Pathfinding failed: parent not found among neighbors");

            reverse_directions.push(graph[edge_index]);
            current_node_index = parent_index;
        }

        reverse_directions.reverse();
        reverse_directions
    };

    Some(directions)
}

#[derive(Copy, Clone, Eq)]
struct OpenListEntry {
    node_index: NodeIndex,
    score: u16,
}

impl PartialEq for OpenListEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl PartialOrd for OpenListEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenListEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.cmp(&self.score)
    }
}

#[derive(Eq, Clone)]
struct BoardNode {
    board: Option<Board>,
    dist_from_initial: Option<u16>,
    heuristic_dist_from_solution: u16,
    parent: Option<NodeIndex>,
    is_in_open_list: bool,
}

impl BoardNode {
    fn new(board: Board, solution_board: &Board) -> Self {
        let heuristic_dist_from_solution = Self::board_heuristic_distance(&board, solution_board);

        Self {
            board: Some(board),
            dist_from_initial: None,
            heuristic_dist_from_solution,
            parent: None,
            is_in_open_list: false,
        }
    }

    pub fn clear_board(&mut self) {
        if self.parent.is_some() {
            self.board = None;
        }
    }

    pub fn get_board<'a>(
        &'a self,
        graph: &DiGraph<BoardNode, Direction>,
        mut node_index: NodeIndex,
    ) -> std::borrow::Cow<'a, Board> {
        if let Some(board) = &self.board {
            return std::borrow::Cow::Borrowed(board);
        }

        let mut moves = Vec::new();
        while graph[node_index].board.is_none() {
            let parent_index = graph[node_index].parent.unwrap_or_else(|| {
                panic!("Node {} has no parent, and no board", node_index.index())
            });

            let edge_index = graph
                .find_edge(parent_index, node_index)
                .unwrap_or_else(|| {
                    panic!(
                        "Parent {} and child {} aren't neighbors",
                        parent_index.index(),
                        node_index.index()
                    )
                });
            moves.push(graph[edge_index]);

            node_index = parent_index;
        }

        let mut board = graph[node_index].board.clone().unwrap();
        for d in moves.into_iter().rev() {
            board.swap_empty_tile(d);
        }

        std::borrow::Cow::Owned(board)
    }

    pub fn f_score(&self) -> u16 {
        self.dist_from_initial
            .map_or(u16::MAX, |dist_from_initial| {
                dist_from_initial + self.heuristic_dist_from_solution
            })
    }

    #[tracing::instrument]
    fn board_heuristic_distance(a: &Board, b: &Board) -> u16 {
        debug_assert_eq!(a.width(), b.width());
        debug_assert_eq!(a.height(), b.height());

        let mut num_to_pos = vec![[0; 2]; a.width() * a.height()];
        for (pos, i) in a.enumerate() {
            num_to_pos[i as usize] = pos;
        }
        b.enumerate()
            .map(|([bx, by], i)| {
                let [ax, ay] = num_to_pos[i as usize];
                ax.abs_diff(bx) + ay.abs_diff(by)
            })
            .sum::<usize>() as u16
    }
}

impl std::fmt::Debug for BoardNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoardNode")
            .field("f_score", &self.f_score())
            .field("dist_from_initial", &self.dist_from_initial)
            .field(
                "heuristic_dist_from_solution",
                &self.heuristic_dist_from_solution,
            )
            .field("is_in_open_list", &self.is_in_open_list)
            .field("parent", &self.parent)
            .field("board", &self.board)
            .finish()
    }
}

impl PartialEq for BoardNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_score() == other.f_score()
    }
}

impl PartialOrd for BoardNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoardNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse comparison to turn BinaryHeap from max-heap to min-heap
        other.f_score().cmp(&self.f_score())
    }
}

#[derive(Clone, Eq, Debug)]
pub struct Board {
    empty_cell_pos: [u8; 2],
    inner: PackedArray2D,
}

impl std::fmt::Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.inner.height() {
            for x in 0..self.inner.width() {
                if x as u8 == self.empty_cell_pos[0] && y as u8 == self.empty_cell_pos[1] {
                    write!(f, "{:>2} ", "☐")?;
                } else {
                    write!(f, "{:>2} ", self.inner.get([x, y]) + 1)?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::ops::Deref for Board {
    type Target = PackedArray2D;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for Board {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Hash for Board {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl Board {
    pub fn new(width: usize, height: usize) -> Self {
        let inner = PackedArray2D::from_slice(
            width,
            height,
            &(0..width * height).map(|i| i as u8).collect::<Vec<_>>(),
        );
        Self::try_from(inner).unwrap()
    }

    pub fn from_human_readable(board: &[&[u8]]) -> Self {
        let height = board.len();
        let width = board[0].len();
        let inner = PackedArray2D::from_slice(
            width,
            height,
            &board
                .iter()
                .flat_map(|&row| row.iter().map(|&v| v - 1))
                .collect::<Vec<_>>(),
        );
        Self::try_from(inner).unwrap()
    }

    pub fn has_solution(&self) -> bool {
        let number_of_swaps_until_sorted =
            number_of_swaps_until_sorted(&mut self.inner.as_1d().to_vec());
        let empty_cell_dist = self.empty_cell_distance_from_solution();

        number_of_swaps_until_sorted % 2 == empty_cell_dist % 2
    }

    fn empty_cell_distance_from_solution(&self) -> usize {
        let [x, y] = self.empty_cell_pos;
        (x as usize).abs_diff(self.inner.width() - 1)
            + (y as usize).abs_diff(self.inner.height() - 1)
    }

    #[tracing::instrument]
    pub fn possible_moves(&self) -> staticvec::StaticVec<Direction, 4> {
        let mut possible_swap_directions = staticvec::StaticVec::<_, 4>::new();

        if self.empty_cell_pos[0] > 0 {
            possible_swap_directions.push(Direction::Left);
        }
        if (self.empty_cell_pos[0] as usize) < self.inner.width() - 1 {
            possible_swap_directions.push(Direction::Right);
        }
        if self.empty_cell_pos[1] > 0 {
            possible_swap_directions.push(Direction::Up);
        }
        if (self.empty_cell_pos[1] as usize) < self.inner.height() - 1 {
            possible_swap_directions.push(Direction::Down);
        }

        possible_swap_directions
    }

    #[tracing::instrument]
    pub fn swap_empty_tile(&mut self, direction: Direction) {
        match direction {
            Direction::Left if self.empty_cell_pos[0] > 0 => {
                self.inner.swap(
                    self.empty_cell_pos,
                    [self.empty_cell_pos[0] - 1, self.empty_cell_pos[1]],
                );
                self.empty_cell_pos[0] -= 1;
            }
            Direction::Right if (self.empty_cell_pos[0] as usize) < self.inner.width() - 1 => {
                self.inner.swap(
                    self.empty_cell_pos,
                    [self.empty_cell_pos[0] + 1, self.empty_cell_pos[1]],
                );
                self.empty_cell_pos[0] += 1;
            }
            Direction::Up if self.empty_cell_pos[1] > 0 => {
                self.inner.swap(
                    self.empty_cell_pos,
                    [self.empty_cell_pos[0], self.empty_cell_pos[1] - 1],
                );
                self.empty_cell_pos[1] -= 1;
            }
            Direction::Down if (self.empty_cell_pos[1] as usize) < self.inner.height() - 1 => {
                self.inner.swap(
                    self.empty_cell_pos,
                    [self.empty_cell_pos[0], self.empty_cell_pos[1] + 1],
                );
                self.empty_cell_pos[1] += 1;
            }
            _ => panic!("Tried to swap {direction} on board:\n{:?}", self.inner),
        }
    }
}

impl TryFrom<PackedArray2D> for Board {
    type Error = BoardError;
    fn try_from(inner: PackedArray2D) -> Result<Self, Self::Error> {
        let mut counts = FnvHashMap::<u8, usize>::default();
        inner.as_1d().iter().for_each(|v| {
            *counts.entry(v).or_default() += 1;
        });
        let empty_cell_pos = inner
            .enumerate()
            .find(|&(_, v)| v as usize == inner.width() * inner.height() - 1)
            .ok_or(BoardError::NoEmptyCell)?
            .0;

        if let Some((&n, _)) = counts.iter().find(|&(_, &c)| c > 1) {
            return Err(BoardError::DuplicateNumber(n));
        }

        for i in 0..inner.width() * inner.height() {
            let i = i as u8;
            if counts.get(&i).is_none() {
                return Err(BoardError::MissingNumber(i));
            }
        }

        Ok(Self {
            empty_cell_pos: [empty_cell_pos[0] as _, empty_cell_pos[1] as _],
            inner,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardError {
    NoEmptyCell,
    DuplicateNumber(u8),
    MissingNumber(u8),
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Direction {
    Left,
    Up,
    Right,
    Down,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Direction::Left => write!(f, "left"),
            Direction::Up => write!(f, "up"),
            Direction::Right => write!(f, "right"),
            Direction::Down => write!(f, "down"),
        }
    }
}

fn hash<T>(value: T) -> u64
where
    T: Hash,
{
    let mut board_hasher = DefaultHasher::default();
    value.hash(&mut board_hasher);
    board_hasher.finish()
}

fn number_of_swaps_until_sorted<T>(arr: &mut [T]) -> usize
where
    T: std::cmp::Ord,
{
    let mut num_swaps = 0;

    loop {
        let mut did_swap = false;

        for i in 0..arr.len() - 1 {
            if arr[i] > arr[i + 1] {
                arr.swap(i, i + 1);
                num_swaps += 1;
                did_swap = true;
            }
        }

        if !did_swap {
            break;
        }
    }

    num_swaps
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn board_swap_left_up() {
        let mut board = Board::new(3, 3);
        board.swap_empty_tile(Direction::Left);

        assert_eq!(
            board,
            Board::from_human_readable(&[&[1, 2, 3], &[4, 5, 6], &[7, 9, 8],])
        );
        assert_eq!(board.empty_cell_distance_from_solution(), 1);

        let mut board = Board::new(3, 3);
        board.swap_empty_tile(Direction::Up);

        assert_eq!(
            board,
            Board::from_human_readable(&[&[1, 2, 3], &[4, 5, 9], &[7, 8, 6],])
        );
        assert_eq!(board.empty_cell_distance_from_solution(), 1);
    }

    #[test]
    fn trivial_board_heuristic() {
        let solution_board = Board::new(3, 3);
        let mut board = solution_board.clone();

        board.swap_empty_tile(Direction::Left);
        assert_eq!(
            BoardNode::board_heuristic_distance(&board, &solution_board),
            2
        );

        board.swap_empty_tile(Direction::Left);
        assert_eq!(
            BoardNode::board_heuristic_distance(&board, &solution_board),
            4
        );

        board.swap_empty_tile(Direction::Right);
        board.swap_empty_tile(Direction::Up);
        assert_eq!(
            BoardNode::board_heuristic_distance(&board, &solution_board),
            4
        );
    }

    #[test]
    fn trivial_board_solve() {
        let mut board = Board::new(3, 3);
        board.swap_empty_tile(Direction::Left);

        assert_eq!(solve_board(board), Some(vec![Direction::Right]));
    }

    #[test]
    fn impossible_board() {
        let mut board = Board::new(4, 4);
        board.set([2, 3], 14);
        board.set([1, 3], 15);

        assert_eq!(solve_board(board), None);
    }

    #[test]
    fn easy_board() {
        let board = Board::from_human_readable(&[&[5, 7, 6], &[1, 9, 3], &[2, 4, 8]]);

        assert_ne!(solve_board(board), None);
    }
}
