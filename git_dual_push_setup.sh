#!/bin/bash
# Script to set up dual push to both Gitee (origin) and GitHub
# while keeping pull from origin (Gitee) only

echo "Setting up dual push configuration..."

# Configure origin to push to both Gitee and GitHub
git remote set-url --add --push origin https://gitee.com/yunjinqi/empyrical.git
git remote set-url --add --push origin https://github.com/cloudQuant/empyrical.git

echo "Configuration complete!"
echo ""
echo "Current remote configuration:"
git remote -v
echo ""
echo "Usage:"
echo "  git push            # Pushes to both Gitee and GitHub"
echo "  git push origin     # Pushes to both Gitee and GitHub"
echo "  git pull            # Pulls from Gitee (origin)"
echo "  git pull origin     # Pulls from Gitee"
echo "  git pull github     # Pulls from GitHub"
echo ""
echo "To push to only one remote:"
echo "  git push github     # Push only to GitHub"
echo "  git push https://gitee.com/yunjinqi/empyrical.git  # Push only to Gitee"