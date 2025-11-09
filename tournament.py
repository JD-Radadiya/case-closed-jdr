import json
import itertools
from typing import Dict, List, Tuple
import pandas as pd
from match_runner import MatchRunner
from model_manager import ModelManager

class Tournament:
    def __init__(self, models_dir: str = "tournament_models"):
        self.model_manager = ModelManager(models_dir=models_dir)
        self.match_runner = MatchRunner(verbose=False)
        
    def load_minimax_versions(self) -> Dict[str, Dict]:
        """Load all minimax model versions"""
        versions = {}
        for version in range(2, 8):  # v1 through v4
            try:
                model_config = self.model_manager.load_model(f'best_minimax_v{version}')
                if model_config:
                    versions[f'v{version}'] = model_config
            except:
                print(f"Could not load minimax v{version}")
        return versions

    def run_tournament(self, num_matches: int = 100) -> pd.DataFrame:
        """Run round-robin tournament between all versions"""
        versions = self.load_minimax_versions()
        results = []
        
        # Generate all possible pairs
        pairs = list(itertools.combinations(versions.keys(), 2))
        
        for v1_name, v2_name in pairs:
            v1_config = versions[v1_name]
            v2_config = versions[v2_name]
            
            print(f"\nMatching {v1_name} vs {v2_name}")
            
            # Run matches in both directions (A vs B and B vs A)
            # First direction - v1 as agent1, v2 as agent2
            forward_results = self._run_match_pair(
                v1_config, v2_config, 
                num_matches=num_matches//2,
                prefix=f"{v1_name}_vs_{v2_name}"
            )
            
            # Reverse direction - v2 as agent1, v1 as agent2
            reverse_results = self._run_match_pair(
                v2_config, v1_config,
                num_matches=num_matches//2,
                prefix=f"{v2_name}_vs_{v1_name}"
            )
            
            # Combine results
            v1_wins = forward_results['agent1_wins'] + reverse_results['agent2_wins']
            v2_wins = forward_results['agent2_wins'] + reverse_results['agent1_wins']
            draws = forward_results['draws'] + reverse_results['draws']
            
            results.append({
                'version1': v1_name,
                'version2': v2_name,
                'v1_wins': v1_wins,
                'v2_wins': v2_wins,
                'draws': draws,
                'v1_win_rate': v1_wins / num_matches,
                'v2_win_rate': v2_wins / num_matches,
                'draw_rate': draws / num_matches,
                'total_matches': num_matches
            })
            
        return pd.DataFrame(results)

    def _run_match_pair(self, 
                       agent1_config: Dict, 
                       agent2_config: Dict,
                       num_matches: int,
                       prefix: str) -> Dict:
        """Run a set of matches between two agents"""
        
        # Ensure agent IDs are set correctly
        agent1_config = {**agent1_config, 'agent_id': 1}
        agent2_config = {**agent2_config, 'agent_id': 2}
        
        results = self.match_runner.run_match_batch(
            agent1_config,
            agent2_config,
            num_matches=num_matches
        )
        
        return results

    def print_tournament_summary(self, results_df: pd.DataFrame):
        """Print readable tournament summary with rankings"""
        # Calculate overall stats per version
        versions = set(results_df['version1'].unique()) | set(results_df['version2'].unique())
        stats = []
        
        for version in versions:
            wins_as_v1 = results_df[results_df['version1'] == version]['v1_wins'].sum()
            wins_as_v2 = results_df[results_df['version2'] == version]['v2_wins'].sum()
            total_matches = (
                results_df[results_df['version1'] == version]['total_matches'].sum() +
                results_df[results_df['version2'] == version]['total_matches'].sum()
            )
            
            win_rate = (wins_as_v1 + wins_as_v2) / total_matches
            
            stats.append({
                'version': version,
                'total_wins': wins_as_v1 + wins_as_v2,
                'total_matches': total_matches,
                'win_rate': win_rate
            })
        
        rankings = pd.DataFrame(stats).sort_values('win_rate', ascending=False)
        
        print("\n=== Tournament Rankings ===")
        print(rankings.to_string(index=False))
        
        print("\n=== Detailed Matchups ===")
        print(results_df.to_string(index=False))

def main():
    tournament = Tournament()
    results = tournament.run_tournament(num_matches=200)  # 100 matches each direction
    tournament.print_tournament_summary(results)
    
    # Save results
    results.to_csv("tournament_results.csv", index=False)
    print("\nResults saved to tournament_results.csv")

if __name__ == "__main__":
    main()