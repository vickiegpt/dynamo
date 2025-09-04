#!/usr/bin/env python3
"""
Script to upload GitHub Actions workflow and job metrics to the dynamo-metrics database.
This script captures job and workflow data including queue time and other relevant metrics.
Database URLs are loaded securely from environment variables.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# Database configuration - URLs loaded from environment variables
PIPELINE_INDEX = os.getenv('PIPELINE_INDEX', '')
JOB_INDEX = os.getenv('JOB_INDEX', '')

class GitHubMetricsUploader:
    def __init__(self):
        self.headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
        self.pipeline_index = PIPELINE_INDEX
        self.jobs_index = JOB_INDEX
        
        # Validate that database URLs are provided
        if not self.pipeline_index or not self.jobs_index:
            raise ValueError(
                "Database URLs not configured. Please set environment variables:\n"
                "  PIPELINE_INDEX - URL for pipeline metrics\n"
                "  JOB_INDEX - URL for job metrics"
            )
        
    def post_to_db(self, url: str, data: Dict[str, Any]) -> None:
        """Push json data to the database/OpenSearch URL"""
        print(f"Posting metrics to database...")
        response = requests.post(url, data=json.dumps(data), headers=self.headers)
        if not (200 <= response.status_code < 300):
            raise ValueError(f"Error posting to DB: {response.status_code}, {response.text}")
        print(f"Successfully posted metrics with ID: {data.get('_id', 'unknown')}")

    def get_github_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from GitHub API"""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("Warning: GITHUB_TOKEN not set, skipping API calls")
            return None
            
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(f"https://api.github.com{endpoint}", headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching GitHub API data from {endpoint}: {e}")
            return None

    def calculate_queue_time(self, created_at: str, started_at: Optional[str]) -> Optional[float]:
        """Calculate queue time in seconds"""
        if not started_at:
            return None
        try:
            created = pd.to_datetime(created_at)
            started = pd.to_datetime(started_at)
            return (started - created).total_seconds()
        except Exception as e:
            print(f"Error calculating queue time: {e}")
            return None

    def post_workflow_metrics(self) -> None:
        """Extract and post workflow (pipeline equivalent) metrics"""
        repo = os.getenv('GITHUB_REPOSITORY')
        run_id = os.getenv('GITHUB_RUN_ID')
        
        if not repo or not run_id:
            print("Missing required environment variables for workflow metrics")
            return
            
        # Get workflow run data from GitHub API
        workflow_data = self.get_github_api_data(f"/repos/{repo}/actions/runs/{run_id}")
        if not workflow_data:
            print("Could not fetch workflow data from GitHub API")
            return
            
        # Extract workflow metrics
        db_data = {}
        db_data["_id"] = f"github_{run_id}_{repo.replace('/', '_')}"
        db_data["s_pipeline_id"] = str(run_id)
        db_data["s_commit_sha"] = workflow_data.get('head_sha', os.getenv('GITHUB_SHA', ''))
        db_data["s_gitlab_branch"] = workflow_data.get('head_branch', os.getenv('GITHUB_REF_NAME', ''))
        db_data["s_pipeline_status"] = workflow_data.get('status', 'unknown')
        db_data["s_pipeline_source"] = workflow_data.get('event', os.getenv('GITHUB_EVENT_NAME', 'unknown'))
        
        # Timestamps
        created_at = workflow_data.get('created_at')
        if created_at:
            db_data["ts_created"] = int(pd.to_datetime(created_at).timestamp() * 1000)
        
        updated_at = workflow_data.get('updated_at')
        if updated_at and workflow_data.get('status') in ['completed', 'cancelled', 'failure']:
            db_data["ts_finished"] = int(pd.to_datetime(updated_at).timestamp() * 1000)
        
        # Duration and queue time
        if created_at and updated_at:
            duration = (pd.to_datetime(updated_at) - pd.to_datetime(created_at)).total_seconds()
            db_data["d_pipeline_duration"] = duration
            
        # Queue time (time from created to run_started_at)
        run_started_at = workflow_data.get('run_started_at')
        if created_at and run_started_at:
            queue_duration = self.calculate_queue_time(created_at, run_started_at)
            if queue_duration is not None:
                db_data["d_pipeline_queue_duration"] = queue_duration
        
        # User and project info
        actor = workflow_data.get('actor', {})
        db_data["s_user_alias"] = actor.get('login', os.getenv('GITHUB_ACTOR', 'unknown'))
        db_data["s_project_id"] = f"github_{repo.replace('/', '_')}"
        db_data["s_project_name"] = repo
        
        # Additional GitHub-specific fields
        db_data["s_workflow_name"] = workflow_data.get('name', os.getenv('GITHUB_WORKFLOW', 'unknown'))
        db_data["s_github_event"] = os.getenv('GITHUB_EVENT_NAME', 'unknown')
        db_data["s_github_ref"] = os.getenv('GITHUB_REF', '')
        
        self.post_to_db(self.pipeline_index, db_data)

    def post_job_metrics(self) -> None:
        """Extract and post job metrics"""
        repo = os.getenv('GITHUB_REPOSITORY')
        run_id = os.getenv('GITHUB_RUN_ID')
        job_name = os.getenv('GITHUB_JOB')
        
        if not repo or not run_id or not job_name:
            print("Missing required environment variables for job metrics")
            return
            
        # Get jobs data from GitHub API
        jobs_data = self.get_github_api_data(f"/repos/{repo}/actions/runs/{run_id}/jobs")
        if not jobs_data or 'jobs' not in jobs_data:
            print("Could not fetch jobs data from GitHub API")
            return
            
        # Find the current job
        current_job = None
        for job in jobs_data['jobs']:
            if job['name'] == job_name:
                current_job = job
                break
                
        if not current_job:
            print(f"Could not find job '{job_name}' in API response")
            return
            
        # Extract job metrics
        db_data = {}
        job_id = current_job['id']
        db_data["_id"] = f"github_{job_id}_{repo.replace('/', '_')}"
        db_data["s_pipeline_id"] = str(run_id)
        db_data["s_pipeline_status"] = current_job.get('status', 'unknown')
        db_data["s_pipeline_source"] = os.getenv('GITHUB_EVENT_NAME', 'unknown')
        db_data["s_job_id"] = str(job_id)
        db_data["s_job_stage"] = "github_actions"  # GitHub Actions doesn't have stages like GitLab
        db_data["s_gitlab_branch"] = os.getenv('GITHUB_REF_NAME', '')
        db_data["s_job_name"] = job_name
        
        # Timestamps
        created_at = current_job.get('created_at')
        if created_at:
            db_data["ts_created"] = int(pd.to_datetime(created_at).timestamp() * 1000)
            
        completed_at = current_job.get('completed_at')
        if completed_at:
            db_data["ts_finished"] = int(pd.to_datetime(completed_at).timestamp() * 1000)
            
        db_data["s_job_status"] = current_job.get('conclusion', current_job.get('status', 'unknown'))
        db_data["s_job_failure_reason"] = current_job.get('conclusion', '-') if current_job.get('conclusion') in ['failure', 'cancelled'] else '-'
        
        # Tags (using labels from runner)
        runner_labels = current_job.get('labels', [])
        db_data["s_tags"] = runner_labels if runner_labels else ['-']
        
        # Duration and queue time
        started_at = current_job.get('started_at')
        if created_at and completed_at:
            duration = (pd.to_datetime(completed_at) - pd.to_datetime(created_at)).total_seconds()
            db_data["d_job_duration"] = duration
            
        # Queue time (time from created to started)
        if created_at and started_at:
            queue_duration = self.calculate_queue_time(created_at, started_at)
            if queue_duration is not None:
                db_data["d_job_queue_duration"] = queue_duration
        
        # Runner info
        runner_name = current_job.get('runner_name')
        runner_id = current_job.get('runner_id')
        db_data["s_runner_id"] = str(runner_id) if runner_id else None
        db_data["s_runner_name"] = runner_name
        
        # User and project info
        db_data["s_user_alias"] = os.getenv('GITHUB_ACTOR', 'unknown')
        db_data["s_project_id"] = f"github_{repo.replace('/', '_')}"
        db_data["s_project_name"] = repo
        
        # Additional GitHub-specific fields
        db_data["s_workflow_name"] = os.getenv('GITHUB_WORKFLOW', 'unknown')
        db_data["s_github_event"] = os.getenv('GITHUB_EVENT_NAME', 'unknown')
        db_data["s_github_ref"] = os.getenv('GITHUB_REF', '')
        
        self.post_to_db(self.jobs_index, db_data)

def main():
    """Main function to upload GitHub Actions metrics"""
    uploader = GitHubMetricsUploader()
    
    # Upload workflow metrics
    try:
        print("Uploading workflow metrics...")
        uploader.post_workflow_metrics()
        print("Workflow metrics uploaded successfully")
    except Exception as e:
        print(f"Error uploading workflow metrics: {e}")
        
    # Upload job metrics
    try:
        print("Uploading job metrics...")
        uploader.post_job_metrics()
        print("Job metrics uploaded successfully")
    except Exception as e:
        print(f"Error uploading job metrics: {e}")

if __name__ == "__main__":
    main()
