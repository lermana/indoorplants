#!/bin/sh

setup_git() {
  git config --global user.email "travis@travis-ci.org"
  git config --global user.name "Travis CI"
}

commit_repo() {
  git add . *
  git commit --message "Travis build: v${INDOORPLANTS_VERSION}"
}

upload_files() {
  if  git ls-remote --exit-code origin > /dev/null 2>&1; then
	  git remote rm origin
  fi

  git remote add origin https://${GH_TOKEN}@github.com/lermana/indoorplants.git
  git push origin master
}

setup_git
commit_repo
upload_files
