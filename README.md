# mat-dp-pipeline

# Standard Data Format

```
World
	tech.csv (coal)
	tech_2016.csv (coal)
	indicators
	-> Africa
		tech.csv (nuclear, wood)
		tech_2015.csv (nuclear, wood) B
		tech_2020.csv (nuclear, wood)
		-> Kenya
			tech.csv (fish, fusion)
			tech_2011.csv (nuclear) A
			tech_2017.csv (wood, coal)
			targets.csv (coal, nuclear)
		-> England
			tech.csv
			targets.csv <<- required file for every country!!!
```
