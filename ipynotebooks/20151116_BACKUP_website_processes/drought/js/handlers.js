$(document).ready(function() {

    // All sides
    var sides = ["left", "top", "right", "bottom"];
    $("h1 span.version").text($.fn.sidebar.version);
	
    // Initialize sidebars
    for (var i = 0; i < sides.length; ++i) {
        var cSide = sides[i];
        $(".sidebar." + cSide).sidebar({
            side: cSide
        });
    }
	
    // Click handlers
    $(".btn[data-action]").on("click", function() {
        var $this = $(this);
        var action = $this.attr("data-action");
        var side = $this.attr("data-side");
        $(".sidebar." + side).trigger("sidebar:" + action);
        return false;
    });
	
    //Toggle attribute
    $('#toggleClick').attr('title', 'Close menu').click(function() {
        $(this).toggleClass('checked');
        var title = 'Close menu';        
        $('.fixed-dialog').attr('style', function(i, s) {
            return (s || '') + 'width: calc(100% - 450px) !important;'
        });
		
		// Adjust charts when menu opens
		if (typeof drought_derived_chart != 'undefined') {
			$('#drought-derived-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#drought-derived-chart').highcharts().reflow();
				}
			});
		};
		
		if (typeof hydrology_chart != 'undefined') {
			$('#hydrology-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#hydrology-chart').highcharts().reflow();
				}
			});
		};			

		if (typeof vegetation_chart != 'undefined') {
			$('#vegetation-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#vegetation-chart').highcharts().reflow();
				}
			});
		};			
		
		if (typeof meteorology_chart != 'undefined') {
			$('#meteorology-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#meteorology-chart').highcharts().reflow();
				}
			});
		};			
		
		if (typeof histogram_chart != 'undefined') {
			$('#histogram-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#histogram-chart').highcharts().reflow();
				}
			});
		};

		if (typeof fracarea_chart != 'undefined') {
			$('#fracarea-chart').animate({
				width: '100%'
			}, {
				duration: 600,
				step: function() {
					$('#fracarea-chart').highcharts().reflow();
				}
			});
		};
	
		
		
		
		
        if ($(this).hasClass('checked')) {
            title = 'Open menu';            
            $('.fixed-dialog').attr('style', function(i, s) {
                return (s || '') +
                    'width: calc(100% - 100px) !important; transition: 200ms;'
            });


		    // Adjust charts when menu close			
            if (typeof drought_derived_chart != 'undefined') {
                $('#drought-derived-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#drought-derived-chart').highcharts().reflow();
                    }
                });
            };
			
            if (typeof hydrology_chart != 'undefined') {
                $('#hydrology-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#hydrology-chart').highcharts().reflow();
                    }
                });
            };			

            if (typeof vegetation_chart != 'undefined') {
                $('#vegetation-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#vegetation-chart').highcharts().reflow();
                    }
                });
            };			
			
            if (typeof meteorology_chart != 'undefined') {
                $('#meteorology-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#meteorology-chart').highcharts().reflow();
                    }
                });
            };			

            if (typeof histogram_chart != 'undefined') {
                $('#histogram-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#histogram-chart').highcharts().reflow();
                    }
                });
            };				

            if (typeof fracarea_chart != 'undefined') {
                $('#fracarea-chart').animate({
                    width: '100%'
                }, {
                    duration: 600,
                    step: function() {
                        $('#fracarea-chart').highcharts().reflow();
                    }
                });
            };	
			
			
        }
        $(this).attr('title', title);
    });
});
