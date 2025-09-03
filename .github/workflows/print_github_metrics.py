#!/usr/bin/env python3
"""
Script to collect and print GitHub Actions workflow and job metrics.
This script captures job and workflow data including queue time and other relevant metrics,
and prints them to stdout for logging/monitoring purposes.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

class GitHubMetricsCollector:
    def __init__(self):
        pass
        
    def get_github_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from GitHub API"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print(f"⚠️  GITHUB_TOKEN not set, using environment variables for {endpoint}")
            return self.get_env_data(endpoint)
            
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            print(f"🔗 Fetching from GitHub API: {endpoint}")
            response = requests.get(f"https://api.github.com{endpoint}", headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"✅ Successfully fetched data from GitHub API")
            return data
        except Exception as e:
            print(f"❌ Error fetching GitHub API data from {endpoint}: {e}")
            print("🔄 Falling back to environment variables")
            return self.get_env_data(endpoint)

    def get_env_data(self, endpoint: str) -> Dict[str, Any]:
        """Generate data from environment variables when API isn't available"""
        repo = os.getenv('GITHUB_REPOSITORY', 'unknown/repo')
        run_id = os.getenv('GITHUB_RUN_ID', '0')
        
        if 'runs/' in endpoint and '/jobs' not in endpoint:
            # Workflow run data from environment
            return {
                'id': int(run_id) if run_id.isdigit() else 0,
                'name': os.getenv('GITHUB_WORKFLOW', 'Unknown Workflow'),
                'head_branch': os.getenv('GITHUB_REF_NAME', 'unknown'),
                'head_sha': os.getenv('GITHUB_SHA', 'unknown'),
                'status': 'in_progress',  # We don't know the final status yet
                'event': os.getenv('GITHUB_EVENT_NAME', 'unknown'),
                'created_at': datetime.now().isoformat() + 'Z',
                'run_started_at': datetime.now().isoformat() + 'Z',
                'actor': {'login': os.getenv('GITHUB_ACTOR', 'unknown')},
                'repository': {'full_name': repo}
            }
        elif '/jobs' in endpoint:
            # Job data from environment
            job_name = os.getenv('GITHUB_JOB', 'unknown-job')
            return {
                'jobs': [{
                    'id': int(os.getenv('GITHUB_RUN_ID', '0')) * 1000,  # Mock job ID
                    'name': job_name,
                    'status': 'in_progress',
                    'created_at': datetime.now().isoformat() + 'Z',
                    'started_at': datetime.now().isoformat() + 'Z',
                    'runner_name': 'github-runner',
                    'labels': ['self-hosted', 'linux']
                }]
            }
        return {}

    def calculate_queue_time(self, created_at: str, started_at: Optional[str]) -> Optional[float]:
        """Calculate queue time in seconds"""
        if not started_at:
            return None
        try:
            created = pd.to_datetime(created_at)
            started = pd.to_datetime(started_at)
            queue_time = (started - created).total_seconds()
            return queue_time
        except Exception as e:
            print(f"❌ Error calculating queue time: {e}")
            return None

    def collect_workflow_metrics(self) -> Dict[str, Any]:
        """Collect workflow (pipeline equivalent) metrics"""
        print("\n🔄 Collecting workflow metrics...")
        
        repo = os.getenv('GITHUB_REPOSITORY')
        run_id = os.getenv('GITHUB_RUN_ID')
        
        if not repo or not run_id:
            print("❌ Missing required environment variables for workflow metrics")
            return {}
            
        # Get workflow run data from GitHub API
        workflow_data = self.get_github_api_data(f"/repos/{repo}/actions/runs/{run_id}")
        if not workflow_data:
            print("❌ Could not fetch workflow data")
            return {}
            
        # Extract workflow metrics
        metrics = {}
        metrics["workflow_id"] = str(run_id)
        metrics["repository"] = repo
        metrics["commit_sha"] = workflow_data.get('head_sha', os.getenv('GITHUB_SHA', ''))
        metrics["branch"] = workflow_data.get('head_branch', os.getenv('GITHUB_REF_NAME', ''))
        metrics["workflow_status"] = workflow_data.get('status', 'unknown')
        metrics["event_type"] = workflow_data.get('event', os.getenv('GITHUB_EVENT_NAME', 'unknown'))
        metrics["workflow_name"] = workflow_data.get('name', os.getenv('GITHUB_WORKFLOW', 'unknown'))
        
        # Timestamps
        created_at = workflow_data.get('created_at')
        if created_at:
            metrics["created_at"] = created_at
        
        updated_at = workflow_data.get('updated_at')
        if updated_at:
            metrics["updated_at"] = updated_at
        
        # Duration and queue time
        if created_at and updated_at:
            try:
                duration = (pd.to_datetime(updated_at) - pd.to_datetime(created_at)).total_seconds()
                metrics["workflow_duration_seconds"] = duration
            except:
                pass
            
        # Queue time (time from created to run_started_at)
        run_started_at = workflow_data.get('run_started_at')
        if created_at and run_started_at:
            queue_duration = self.calculate_queue_time(created_at, run_started_at)
            if queue_duration is not None:
                metrics["workflow_queue_time_seconds"] = queue_duration
        
        # User info
        actor = workflow_data.get('actor', {})
        metrics["triggered_by"] = actor.get('login', os.getenv('GITHUB_ACTOR', 'unknown'))
        
        return metrics

    def collect_job_metrics(self) -> Dict[str, Any]:
        """Collect job metrics"""
        print("\n🔄 Collecting job metrics...")
        
        repo = os.getenv('GITHUB_REPOSITORY')
        run_id = os.getenv('GITHUB_RUN_ID')
        job_name = os.getenv('GITHUB_JOB')
        
        if not repo or not run_id or not job_name:
            print("❌ Missing required environment variables for job metrics")
            return {}
            
        # Get jobs data from GitHub API
        jobs_data = self.get_github_api_data(f"/repos/{repo}/actions/runs/{run_id}/jobs")
        if not jobs_data or 'jobs' not in jobs_data:
            print("❌ Could not fetch jobs data")
            return {}
            
        # Find the current job
        current_job = None
        for job in jobs_data['jobs']:
            if job['name'] == job_name:
                current_job = job
                break
                
        if not current_job:
            print(f"❌ Could not find job '{job_name}' in API response")
            available_jobs = [job['name'] for job in jobs_data['jobs']]
            print(f"Available jobs: {available_jobs}")
            return {}
            
        print(f"✅ Found job: {current_job['name']}")
        
        # Extract job metrics
        metrics = {}
        metrics["job_id"] = str(current_job['id'])
        metrics["job_name"] = job_name
        metrics["workflow_id"] = str(run_id)
        metrics["repository"] = repo
        metrics["branch"] = os.getenv('GITHUB_REF_NAME', '')
        
        # Timestamps
        created_at = current_job.get('created_at')
        if created_at:
            metrics["created_at"] = created_at
            
        started_at = current_job.get('started_at')
        if started_at:
            metrics["started_at"] = started_at
            
        completed_at = current_job.get('completed_at')
        if completed_at:
            metrics["completed_at"] = completed_at
            
        metrics["job_status"] = current_job.get('conclusion', current_job.get('status', 'unknown'))
        
        # Runner info
        runner_labels = current_job.get('labels', [])
        metrics["runner_labels"] = runner_labels
        metrics["runner_name"] = current_job.get('runner_name', 'unknown')
        
        # Duration and queue time
        if created_at and completed_at:
            try:
                duration = (pd.to_datetime(completed_at) - pd.to_datetime(created_at)).total_seconds()
                metrics["job_duration_seconds"] = duration
            except:
                pass
            
        # Queue time (time from created to started)
        if created_at and started_at:
            queue_duration = self.calculate_queue_time(created_at, started_at)
            if queue_duration is not None:
                metrics["job_queue_time_seconds"] = queue_duration
        
        # User info
        metrics["triggered_by"] = os.getenv('GITHUB_ACTOR', 'unknown')
        
        return metrics

    def print_metrics(self, workflow_metrics: Dict[str, Any], job_metrics: Dict[str, Any]) -> None:
        """Print collected metrics in a readable format"""
        print("\n" + "="*80)
        print("📊 GITHUB ACTIONS METRICS SUMMARY")
        print("="*80)
        
        if workflow_metrics:
            print("\n🔄 WORKFLOW METRICS:")
            print("-" * 40)
            for key, value in workflow_metrics.items():
                print(f"  {key}: {value}")
                
        if job_metrics:
            print("\n⚙️  JOB METRICS:")
            print("-" * 40)
            for key, value in job_metrics.items():
                print(f"  {key}: {value}")
        
        # Highlight key performance metrics
        print("\n⏱️  KEY PERFORMANCE INDICATORS:")
        print("-" * 40)
        
        if workflow_metrics.get('workflow_queue_time_seconds'):
            print(f"  Workflow Queue Time: {workflow_metrics['workflow_queue_time_seconds']:.2f} seconds")
        if workflow_metrics.get('workflow_duration_seconds'):
            print(f"  Workflow Duration: {workflow_metrics['workflow_duration_seconds']:.2f} seconds")
        if job_metrics.get('job_queue_time_seconds'):
            print(f"  Job Queue Time: {job_metrics['job_queue_time_seconds']:.2f} seconds")
        if job_metrics.get('job_duration_seconds'):
            print(f"  Job Duration: {job_metrics['job_duration_seconds']:.2f} seconds")
            
        print("\n" + "="*80)

def main():
    """Main function to collect and print GitHub Actions metrics"""
    print("🚀 GitHub Actions Metrics Collector")
    print("=" * 50)
    
    collector = GitHubMetricsCollector()
    
    # Collect workflow metrics
    workflow_metrics = {}
    try:
        workflow_metrics = collector.collect_workflow_metrics()
        if workflow_metrics:
            print("✅ Workflow metrics collected successfully")
        else:
            print("⚠️  No workflow metrics collected")
    except Exception as e:
        print(f"❌ Error collecting workflow metrics: {e}")
        
    # Collect job metrics
    job_metrics = {}
    try:
        job_metrics = collector.collect_job_metrics()
        if job_metrics:
            print("✅ Job metrics collected successfully")
        else:
            print("⚠️  No job metrics collected")
    except Exception as e:
        print(f"❌ Error collecting job metrics: {e}")
    
    # Print all metrics
    collector.print_metrics(workflow_metrics, job_metrics)
    
    # Also output as JSON for potential parsing by other tools
    if workflow_metrics or job_metrics:
        print("\n📋 JSON OUTPUT (for parsing):")
        print("-" * 40)
        json_output = {
            "workflow_metrics": workflow_metrics,
            "job_metrics": job_metrics,
            "collection_timestamp": datetime.now().isoformat()
        }
        print(json.dumps(json_output, indent=2, default=str))

if __name__ == "__main__":
    main()
