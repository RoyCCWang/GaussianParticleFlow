

function testBMimplementation(N_samples = 4000)

    t_array = sort(rand(N_samples).*5)
    Bt_array, Z_array = getallBrownianmotions(t_array)

    Bt2_array = Vector{Float64}(undef,N_samples)
    Bt2_array[1] = getnextBrownianmotion(1, 0.0, t_array[1], 0.0, Z_array[1])
    for i = 2:N_samples
        Bt2_array[i] = getnextBrownianmotion(i, t_array[i-1], t_array[i], Bt2_array[i-1], Z_array[i])
    end

    diff = norm(Bt_array - Bt2_array)
    println("diff between incremental and batch implementation of BM is ", diff)

    return nothing
end
