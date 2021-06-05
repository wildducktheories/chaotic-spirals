docker-build:
	docker-compose build spirals

docker-run: docker-build
	docker-compose up -d spirals

docker-token:
	@docker-compose exec -T spirals jupyter notebook list | sed -n "s/.*token=\([^ ]*\) .*/\1/p"

local-build:
	pip3 install -r requirements.txt

local-run:
	jupyter notebook