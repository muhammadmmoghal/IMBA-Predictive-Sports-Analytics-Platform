export interface Player {
  player_name: string;
  division: string;
  team: string;
  confidence_level: 'Low' | 'Medium' | 'High';
  confidence_score: number;
  games_played_history: number;
  predicted_pts: number;
  predicted_reb: number;
  predicted_ast: number;
  predicted_stl: number;
  predicted_blk: number;
  pts_low: number;
  pts_high: number;
  reb_low: number;
  reb_high: number;
  ast_low: number;
  ast_high: number;
  stl_low: number;
  stl_high: number;
  blk_low: number;
  blk_high: number;
  prob_10_plus_pts: number;
  prob_15_plus_pts: number;
  prob_20_plus_pts: number;
  prob_5_plus_reb: number;
  prob_10_plus_reb: number;
  prob_5_plus_ast: number;
  prob_double_double: number;
  // Only in full dataset
  player_id?: string;
  tier?: string;
}

export type DivisionFilter = 'all' | 'comp' | 'rec';
export type LeaderboardTab = 'scorers' | 'rebounders' | 'assists' | 'double_double';
