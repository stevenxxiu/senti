#!/usr/bin/perl
#
#  Author: Preslav Nakov
#  
#  Description: Scores SemEval-2015 task 10, subtask D
#
#  Last modified: December 30, 2014
#
#

use warnings;
use strict;
use utf8;
binmode (STDIN,  ":utf8");
binmode (STDOUT, ":utf8");

my $INPUT_FILE         =  $ARGV[0];
use constant GOLD_FILE => 'SemEval2015-task10-test-D-gold.txt';
my $OUTPUT_FILE        =  $INPUT_FILE . '.scored';


########################
###   MAIN PROGRAM   ###
########################

my $totalAbsDiff      = 0.0;
my $totalAbsLevelDiff = 0;

### 1. Read the files and get the statistics
open INPUT, '<:encoding(UTF-8)', $INPUT_FILE or die;
open GOLD,  '<:encoding(UTF-8)', GOLD_FILE or die;

my $lineNo = 1;
for (; <INPUT>; $lineNo++) {
	s/^[ \t]+//;
	s/[ \t\n\r]+$//;

	### 1.1. Check the input file format
	# aaron rodgers	0.714285714285714
	die "Wrong file format: $_" if (!/^([a-z ]+[^\t ]) ?\t([0-9\.]+)$/);
	my ($propTopic, $propScore) = ($1, $2);

	### 1.2	. Check the gold file format
	# aaron rodgers	0.616
	$_ = <GOLD>;
	s/[\n\r]+$//;
	die "Wrong file format: $_" if (!/^([a-z ]+)\t([0-9\.]+)$/);
	my ($goldTopic, $goldScore) = ($1, $2);

	### 1.3. Make sure topics match
	die "Topic mismatch on line $lineNo: '$propTopic' in $INPUT_FILE, but '$goldTopic' in " . GOLD_FILE if ($propTopic ne $goldTopic);

	### 1.4. Check the score values
	die "Wrong value for the score: $propScore" if (($propScore < 0.0) || ($propScore > 1.0));
	die "Wrong value for the score: $goldScore" if (($goldScore < 0.0) || ($goldScore > 1.0));

	### 1.5. Update the total absolute difference
	$totalAbsDiff += abs($propScore - $goldScore);

	### 1.6. Map the values to categories
	my $propLevel = &getLevel($propScore);
	my $goldLevel = &getLevel($goldScore);
	$totalAbsLevelDiff += abs($propLevel - $goldLevel);

}

close(INPUT) or die;
close(GOLD) or die;


### 2. One last check
$lineNo--;
die "Too few lines in $INPUT_FILE: $lineNo" if (137 != $lineNo);


### 3. Output the results
open OUTPUT, '>:encoding(UTF-8)', $OUTPUT_FILE or die;
my $avgAbsDiff      = $totalAbsDiff / $lineNo;
my $avgAbsLevelDiff = $totalAbsLevelDiff / $lineNo;

print OUTPUT "avgAbsDiff\t$avgAbsDiff\n";
print OUTPUT "avgAbsLevelDiff\t$avgAbsLevelDiff\n";
print "$INPUT_FILE\tTwitter2015-topic-D\t$avgAbsDiff\t$avgAbsLevelDiff\n";
close(OUTPUT) or die;


################
###   SUBS   ###
################

sub getLevel()
{	my $score = shift;
	# strongly positive: 80% < POS/(POS+NEG) <= 100%
	# weakly positive:   60% < POS/(POS+NEG) <= 80%
	# mixed:             40% < POS/(POS+NEG) <= 60%
	# weakly negative:   20% < POS/(POS+NEG) <= 40%
	# strongly negative:  0% <=POS/(POS+NEG) <= 20%
	die "Wrong value for the score: $score" if ($score < 0.0);
	if    ($score <= 0.20) { return 1; }
	elsif ($score <= 0.40) { return 2; }
	elsif ($score <= 0.60) { return 3; }
	elsif ($score <= 0.80) { return 4; }
	elsif ($score <= 1.00) { return 5; }
	die "Wrong value for the score: $score";
}