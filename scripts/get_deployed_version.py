import os
import sys
import replicate

def main():
    """
    Fetch the latest deployed version of a model from Replicate
    and write it to GitHub Actions output.
    """
    model_name = os.environ.get('MODEL_NAME')
    api_token = os.environ.get('REPLICATE_API_TOKEN')
    github_output = os.environ.get('GITHUB_OUTPUT')
    
    if not model_name:
        print("❌ MODEL_NAME environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    if not api_token:
        print("❌ REPLICATE_API_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    try:
        print(f"🔎 Capturing deployed version for {model_name}")
        
        client = replicate.Client(api_token=api_token)
        username = "paragekbote"
        
        # Get the model and its versions
        model = client.models.get(f"{username}/{model_name}")
        versions = list(model.versions.list())
        
        if not versions:
            print(f"❌ No versions found for {username}/{model_name}", file=sys.stderr)
            sys.exit(1)
        
        # Get the latest version ID
        version_id = versions[0].id
        model_ref = f"{username}/{model_name}:{version_id}"
        
        # Write to GitHub Actions output
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"candidate_model_id={model_ref}\n")
        
        print(f"✅ Candidate model: {model_ref}")
        
    except Exception as e:
        print(f"❌ Failed to resolve deployed version: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()