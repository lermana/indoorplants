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
  git remote add origin https://lermana:${GH_TOKEN}@github.com/lermana/indoorplants.git
  git push origin master
}

setup_git
commit_website_files
upload_files
