(function($) {
    $("#tabs").tabs();
    $(function() {
        $("#accordion_timeseries > div").accordion({
            header: "h3",
            heightStyle: "content",
            active: false,
            collapsible: true,
            activate: function(event, ui) {
                var clicked = $(this).find('.ui-state-active').attr('id');
                if ($.trim($('#' + clicked + '_container').text()).length ==
                    0) {
                    $('#' + clicked + '_container').load('tab_ts/' +
                        clicked + '.html')
                };
            }
        });
    });
    $(function() {
        $("#accordion_maps > div").accordion({
            header: "h3",
            heightStyle: "content",
            active: false,
            collapsible: true,
            activate: function(event, ui) {
                var clicked = $(this).find('.ui-state-active').attr('id');
                if ($.trim($('#' + clicked + '_container').text()).length ==
                    0) {
                    $('#' + clicked + '_container').load('tab_maps/' +
                        clicked + '.html')
                };
            }
        });
    });
    $(function() {
        $("#accordion_order > div").accordion({
            header: "h3",
            heightStyle: "content",
            active: false,
            collapsible: true,
            activate: function(event, ui) {
                var clicked = $(this).find('.ui-state-active').attr('id');
                if ($.trim($('#' + clicked + '_container').text()).length ==
                    0) {
                    $('#' + clicked + '_container').load('tab_order/' +
                        clicked + '.html')
                };
            }
        });
    });
})
(jQuery);