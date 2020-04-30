

function visualize2Ddensities(f_array::Vector{Function},
                                x_ranges::Vector{LinRange{T}},
                                X,
                                fig_num::Int,
                                show_markers_flag::Bool,
                                X_markers,
                                marker_style::String,
                                base_title_string) where T <: Real
    #
    N = length(f_array)

    for i = 1:N
        #
        title_string = @sprintf("%s, subset %d",
                                    base_title_string, i)

        f_X = f_array[i].(X)
        fig_num = Utilities.visualizemeshgridpcolor(x_ranges,
                            f_X, X_markers[i], marker_style,
                            fig_num, title_string)
    end
    return fig_num

end
